/*
 * Enhanced Node.js Express application for generating Salesforce activity summaries using OpenAI Assistants.
 *
 * V2 Features:
 * - DUAL-MODE PROCESSING:
 *   - 'asynchronous' (default): Full, parallelized background processing for all historical data.
 *   - 'synchronous': Instant generation of the most recent month/quarter summary, returned in the API response.
 * - PARALLEL EXECUTION: Asynchronous mode uses Promise.all to run OpenAI calls concurrently, drastically reducing total time.
 * - DYNAMIC SOQL: Synchronous mode modifies SOQL to fetch only recent data for speed.
 * - Creates/Retrieves OpenAI Assistants on startup using environment variable IDs as preference.
 * - Salesforce integration (fetching activities, saving summaries).
 * - OpenAI Assistants API V2 usage.
 * - Dynamic function schema loading (default or from request).
 * - Dynamic tool_choice to force specific function calls (monthly/quarterly).
 * - Conditional input method: Direct JSON in prompt (< threshold) or File Upload (>= threshold).
 * - Robust error handling and callback mechanism for async mode.
 * - Temporary file management.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra"); // Using fs-extra for promise-based file operations and JSON handling
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// --- Assistant IDs are now PREFERRED, creation is fallback ---
const OPENAI_MONTHLY_ASSISTANT_ID_ENV = process.env.OPENAI_MONTHLY_ASSISTANT_ID;
const OPENAI_QUARTERLY_ASSISTANT_ID_ENV = process.env.OPENAI_QUARTERLY_ASSISTANT_ID;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c"; // Salesforce object API name
const DIRECT_INPUT_THRESHOLD = 2000; // Max activities for direct JSON input in prompt
const PROMPT_LENGTH_THRESHOLD = 256000; // Character limit for direct prompt input
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files'); // Directory for temporary files

// --- NEW: Constants for processing modes ---
const PROCESSING_MODE_SYNC = 'synchronous';
const PROCESSING_MODE_ASYNC = 'asynchronous';

// --- Environment Variable Validation (Essential Vars) ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Global Variables for Final Assistant IDs ---
let monthlyAssistantId = null;
let quarterlyAssistantId = null;


// --- Default OpenAI Function Schemas (No changes here) ---
const defaultFunctions = [
    {
      "name": "generate_monthly_activity_summary",
      "description": "Generates a structured monthly sales activity summary with insights and categorization based on provided activity data. Apply sub-theme segmentation within activityMapping.",
      "parameters": {
        "type": "object",
        "properties": {
          "summary": {
            "type": "string",
            "description": "HTML summary for the month. MUST have one H1 header 'Sales Activity Summary for {Month} {Year}' (no bold) followed by a UL list of key insights."
          },
          "activityMapping": {
            "type": "object",
            "description": "Activities categorized under predefined themes. Each category key holds an array where each element represents a distinct sub-theme identified within that category.",
            "properties": {
              "Key Themes of Customer Interaction": {
                "type": "array",
                "description": "An array where each element represents a distinct sub-theme identified in customer interactions (e.g., 'Pricing', 'Support'). Generate multiple elements if multiple distinct themes are found.",
                "items": {
                  "type": "object",
                  "description": "Represents a single, specific sub-theme identified within 'Key Themes'. Contains a focused summary and ONLY the activities related to this sub-theme.",
                  "properties": {
                    "Summary": {
                      "type": "string",
                      "description": "A concise summary describing this specific sub-theme ONLY (e.g., 'Discussions focused on contract renewal terms')."
                    },
                    "ActivityList": {
                      "type": "array",
                      "description": "A list containing ONLY the activities specifically relevant to this sub-theme.",
                      "items": {
                        "type": "object",
                        "properties": {
                          "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                          "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from CreatedDate, Subject, Description." },
                          "CreatedDate": { "type": "string", "description": "CreatedDate in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                        },
                        "required": ["Id", "LinkText", "CreatedDate"],
                        "additionalProperties": false
                      }
                    }
                  },
                  "required": ["Summary", "ActivityList"],
                  "additionalProperties": false
                }
              },
              "Tone and Purpose of Interaction": {
                "type": "array",
                "description": "An array where each element represents a distinct tone or strategic intent identified (e.g., 'Information Gathering', 'Negotiation'). Generate multiple elements if distinct patterns are found.",
                 "items": {
                  "type": "object",
                  "description": "Represents a single, specific tone/purpose pattern. Contains a focused summary and ONLY the activities exhibiting this pattern.",
                  "properties": {
                     "Summary": { "type": "string", "description": "A concise summary describing this specific tone/purpose ONLY." },
                     "ActivityList": {
                       "type": "array",
                       "description": "A list containing ONLY the activities specifically exhibiting this tone/purpose.",
                       "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from CreatedDate, Subject, Description." },
                            "CreatedDate": { "type": "string", "description": "CreatedDate in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "CreatedDate"],
                          "additionalProperties": false
                        }
                      }
                   },
                   "required": ["Summary", "ActivityList"], "additionalProperties": false
                 }
              },
              "Recommended Action and Next Steps": {
                "type": "array",
                "description": "An array where each element represents a distinct type of recommended action or next step identified (e.g., 'Schedule Follow-up Demo', 'Send Proposal'). Generate multiple elements if distinct recommendations are found.",
                 "items": {
                   "type": "object",
                   "description": "Represents a single, specific recommended action type. Contains a focused summary and ONLY the activities leading to this recommendation.",
                   "properties": {
                      "Summary": { "type": "string", "description": "A concise summary describing this specific recommendation type ONLY." },
                      "ActivityList": {
                        "type": "array",
                        "description": "A list containing ONLY the activities specifically related to this recommendation.",
                        "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific activity" },
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from CreatedDate, Subject, Description." },
                            "CreatedDate": { "type": "string", "description": "CreatedDate in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "CreatedDate"],
                          "additionalProperties": false
                        }
                      }
                   },
                   "required": ["Summary", "ActivityList"], "additionalProperties": false
                 }
              }
            },
            "required": ["Key Themes of Customer Interaction", "Tone and Purpose of Interaction", "Recommended Action and Next Steps"]
          },
          "activityCount": {
            "type": "integer",
            "description": "Total number of activities processed for the month (matching the input count)."
          }
        },
        "required": ["summary", "activityMapping", "activityCount"]
      }
    },
    {
      "name": "generate_quarterly_activity_summary",
      "description": "Aggregates provided monthly summaries (as JSON) into a structured quarterly report for a specific quarter, grouped by year.",
      "parameters": {
        "type": "object",
        "properties": {
          "yearlySummary": {
            "type": "array",
            "description": "Quarterly summary data, grouped by year. Should typically contain only one year based on input.",
            "items": {
              "type": "object",
              "properties": {
                "year": {
                  "type": "integer",
                  "description": "The calendar year of the quarter being summarized."
                },
                "quarters": {
                  "type": "array",
                  "description": "List containing the summary for the single quarter being processed.",
                  "items": {
                    "type": "object",
                    "properties": {
                      "quarter": {
                        "type": "string",
                        "description": "Quarter identifier (e.g., Q1, Q2, Q3, Q4) corresponding to the input monthly data."
                      },
                      "summary": {
                        "type": "string",
                        "description": "HTML summary for the quarter. MUST have one H1 header 'Sales Activity Summary for {Quarter} {Year}' (no bold) followed by a UL list of key aggregated insights."
                      },
                      "activityMapping": {
                        "type": "array",
                        "description": "Aggregated activities categorized under predefined themes for the entire quarter.",
                        "items": {
                          "type": "object",
                          "description": "Represents one main category for the quarter, containing an aggregated summary and a consolidated list of all relevant activities.",
                          "properties": {
                            "category": {
                              "type": "string",
                              "description": "Category name (Must be one of 'Key Themes of Customer Interaction', 'Tone and Purpose of Interaction', 'Recommended Action and Next Steps')."
                            },
                            "summary": {
                              "type": "string",
                              "description": "Aggregated summary synthesizing findings for this category across the entire quarter, highlighting key quarterly sub-themes identified."
                            },
                            "activityList": {
                              "type": "array",
                              "description": "Consolidated list of ALL activities for this category from the input monthly summaries for this quarter.",
                              "items": {
                                "type": "object",
                                "properties": {
                                  "id": { "type": "string", "description": "Salesforce Activity ID (copied from monthly input)." },
                                  "linkText": { "type": "string", "description": "'MMM DD YYYY: Short Description' (copied from monthly input)." },
                                  "CreatedDate": { "type": "string", "description": "CreatedDate in 'YYYY-MM-DD' format (copied from monthly input)." }
                                },
                                "required": ["id", "linkText", "CreatedDate"],
                                "additionalProperties": false
                              }
                            }
                          },
                          "required": ["category", "summary", "activityList"],
                          "additionalProperties": false
                        }
                      },
                      "activityCount": {
                        "type": "integer",
                        "description": "Total number of unique activities aggregated for the quarter from monthly inputs."
                      },
                      "startdate": {
                        "type": "string",
                        "description": "Start date of the quarter being summarized (YYYY-MM-DD)."
                      }
                    },
                    "required": ["quarter", "summary", "activityMapping", "activityCount", "startdate"]
                  }
                }
              },
              "required": ["year", "quarters"]
            }
          }
        },
        "required": ["yearlySummary"]
      }
    }
];

// --- OpenAI Client Initialization ---
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// --- Express Application Setup ---
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// --- Helper Function to Create or Retrieve Assistant (No changes here) ---
async function createOrRetrieveAssistant(
    openaiClient,
    assistantIdEnvVar,
    assistantName,
    assistantInstructions,
    assistantToolsConfig,
    assistantModel
) {
    if (assistantIdEnvVar) {
        console.log(`Attempting to retrieve Assistant "${assistantName}" using ID: ${assistantIdEnvVar}`);
        try {
            const retrievedAssistant = await openaiClient.beta.assistants.retrieve(assistantIdEnvVar);
            console.log(`Successfully retrieved existing Assistant "${retrievedAssistant.name}" with ID: ${retrievedAssistant.id}`);
            return retrievedAssistant.id;
        } catch (error) {
            if (error instanceof NotFoundError) {
                console.warn(`Assistant with ID "${assistantIdEnvVar}" not found. Will proceed to create a new one for "${assistantName}".`);
            } else {
                console.error(`Error retrieving Assistant "${assistantName}" (ID: ${assistantIdEnvVar}):`, error);
                throw new Error(`Failed to retrieve Assistant ${assistantName}: ${error.message}`);
            }
        }
    } else {
        console.log(`No environment variable ID provided for Assistant "${assistantName}". Proceeding to create a new one.`);
    }

    console.log(`Creating new Assistant: ${assistantName}...`);
    try {
        const newAssistant = await openaiClient.beta.assistants.create({
            name: assistantName,
            instructions: assistantInstructions,
            tools: assistantToolsConfig,
            model: assistantModel,
        });
        console.log(`Successfully created new Assistant "${newAssistant.name}" with ID: ${newAssistant.id}`);
        const envVarName = `OPENAI_${assistantName.toUpperCase().replace(/ /g, '_').replace('SALESFORCE_', '')}_ASSISTANT_ID`;
        console.warn(`--> IMPORTANT: Consider adding this ID to your .env file as ${envVarName}=${newAssistant.id} for future reuse.`);
        return newAssistant.id;
    } catch (creationError) {
        console.error(`Error creating Assistant "${assistantName}":`, creationError);
        throw new Error(`Failed to create Assistant ${assistantName}: ${creationError.message}`);
    }
}

// --- Server Startup (No changes here) ---
(async () => {
    try {
        console.log("Initializing Assistants...");
        await fs.ensureDir(TEMP_FILE_DIR);

        const assistantBaseTools = [
             { type: "file_search" },
             { type: "function" }
        ];

        monthlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_MONTHLY_ASSISTANT_ID_ENV,
            "Salesforce Monthly Summarizer",
            "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Apply sub-theme segmentation within the activityMapping as described in the function schema. Focus on extracting key themes, tone, and recommended actions. Use file_search if data is provided as a file.",
            assistantBaseTools,
            OPENAI_MODEL
        );

        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_QUARTERLY_ASSISTANT_ID_ENV,
            "Salesforce Quarterly Summarizer",
            "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary for a specific quarter using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately based on the input monthly summaries.",
             assistantBaseTools,
             OPENAI_MODEL
        );

        if (!monthlyAssistantId || !quarterlyAssistantId) {
             throw new Error("Failed to obtain valid IDs for one or more Assistants during startup.");
        }

        app.listen(PORT, () => {
            console.log("----------------------------------------------------");
            console.log(`Server running on port ${PORT}`);
            console.log(`Using OpenAI Model (for new Assistants): ${OPENAI_MODEL}`);
            console.log(`Using Monthly Assistant ID: ${monthlyAssistantId}`);
            console.log(`Using Quarterly Assistant ID: ${quarterlyAssistantId}`);
            console.log("Server is ready to accept requests.");
            console.log("----------------------------------------------------");
        });

    } catch (startupError) {
        console.error("FATAL STARTUP ERROR:", startupError.message);
        process.exit(1);
    }
})();


// --- MODIFIED: Main API Endpoint with Dual-Mode Logic ---
app.post('/generatesummary', async (req, res) => {
    console.log("Received /generatesummary request");

    if (!monthlyAssistantId || !quarterlyAssistantId) {
        console.error("Error: Assistants not initialized properly during startup.");
        return res.status(503).json({ error: "Service Unavailable: Assistants are not ready." });
    }

    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        console.warn("Unauthorized request: Missing or invalid Bearer token.");
        return res.status(401).json({ error: "Unauthorized" });
    }
    const accessToken = authHeader.split(" ")[1];

    const {
        accountId,
        callbackUrl,
        userPrompt,
        userPromptQtr,
        queryText,
        summaryMap,
        loggedinUserId,
        sendCallback,
        qtrJSON,
        monthJSON,
        processingMode = PROCESSING_MODE_ASYNC // --- NEW: Default to async
    } = req.body;

    if (!accountId || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId) {
        return res.status(400).send({ error: "Missing required parameters (accountId, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId)" });
    }
     if (processingMode === PROCESSING_MODE_ASYNC && !callbackUrl) {
        return res.status(400).send({ error: "Missing required parameter 'callbackUrl' for asynchronous processing mode." });
    }

    let summaryRecordsMap = {};
    let monthlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_monthly_activity_summary');
    let quarterlyFuncSchema = defaultFunctions.find(f => f.name === 'generate_quarterly_activity_summary');

    try {
        if (summaryMap) summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        if (monthJSON) monthlyFuncSchema = JSON.parse(monthJSON);
        if (qtrJSON) quarterlyFuncSchema = JSON.parse(qtrJSON);
    } catch (e) {
        console.error("Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided. ${e.message}` });
    }

    if (!monthlyFuncSchema || !quarterlyFuncSchema) {
        console.error("FATAL: Function schemas could not be loaded.");
        return res.status(500).send({ error: "Internal server error: Could not load function schemas."});
    }

    // --- NEW: Route logic based on processingMode ---
    if (processingMode === PROCESSING_MODE_SYNC) {
        // --- SYNCHRONOUS PATH ---
        console.log(`[${accountId}] Starting SYNCHRONOUS processing.`);
        try {
            const result = await generateRecentSummary(
                accountId, accessToken, userPrompt, userPromptQtr, queryText, summaryRecordsMap, loggedinUserId,
                monthlyFuncSchema, quarterlyFuncSchema, monthlyAssistantId, quarterlyAssistantId
            );
            console.log(`[${accountId}] Synchronous processing complete. Sending 200 OK with data.`);
            res.status(200).json({ status: 'completed', data: result });
        } catch (error) {
            console.error(`[${accountId}] Error during synchronous processing:`, error);
            res.status(500).json({ error: "Failed to generate recent summary", details: error.message });
        }
    } else {
        // --- ASYNCHRONOUS PATH ---
        console.log(`[${accountId}] Starting ASYNCHRONOUS processing.`);
        res.status(202).json({ status: 'processing', message: 'Full summary generation initiated. You will receive a callback.' });

        processSummary(
            accountId, accessToken, callbackUrl, userPrompt, userPromptQtr, queryText, summaryRecordsMap, loggedinUserId,
            monthlyFuncSchema, quarterlyFuncSchema, monthlyAssistantId, quarterlyAssistantId, sendCallback
        ).catch(async (error) => {
            console.error(`[${accountId}] Unhandled error during async background processing:`, error);
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled processing error: ${error.message}`);
        });
    }
});

// --- Helper Function to Get Quarter from Month Index ---
function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex <= 2) return 'Q1';
    if (monthIndex <= 5) return 'Q2';
    if (monthIndex <= 8) return 'Q3';
    return 'Q4';
}


// --- MODIFIED: Asynchronous Summary Processing Logic with PARALLELISM ---
async function processSummary(
    accountId, accessToken, callbackUrl, userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId, finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId, sendCallback
) {
    console.log(`[${accountId}] Starting parallel background processing...`);
    const conn = new jsforce.Connection({ instanceUrl: SF_LOGIN_URL, accessToken: accessToken, maxRequest: 5, version: '59.0' });

    try {
        // 1. Fetch ALL Salesforce Records
        const groupedData = await fetchRecords(conn, queryText);
        console.log(`[${accountId}] Fetched and grouped all historical data.`);

        // 2. --- PARALLEL: Generate Monthly Summaries ---
        console.log(`[${accountId}] Preparing parallel generation of monthly summaries...`);
        const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
        const monthlyPromises = [];

        for (const year in groupedData) {
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    const activities = monthObj[month];
                    if (activities.length === 0) continue;

                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);
                    const promise = generateSummary(
                        activities, openai, finalMonthlyAssistantId, userPromptMonthly, finalMonthlyFuncSchema
                    ).then(aiOutput => ({ // Return a contextual object
                        year: parseInt(year),
                        month,
                        monthIndex: monthMap[month.toLowerCase()],
                        aiOutput,
                        count: activities.length,
                        startdate: new Date(Date.UTC(year, monthMap[month.toLowerCase()], 1)).toISOString().split('T')[0]
                    })).catch(error => {
                        console.error(`[${accountId}] Error generating summary for ${month} ${year}:`, error.message);
                        return null; // Return null on failure to not break Promise.all
                    });
                    monthlyPromises.push(promise);
                }
            }
        }

        const monthlyResults = (await Promise.all(monthlyPromises)).filter(Boolean); // Await all and filter out nulls
        console.log(`[${accountId}] Parallel monthly generation complete. ${monthlyResults.length} summaries created.`);

        // 3. Process and Save Monthly Summaries
        const finalMonthlySummaries = {}; // { year: { month: { ... } } }
        const monthlyForSalesforce = {};  // { year: { month: { ... } } }

        monthlyResults.forEach(result => {
            if (!finalMonthlySummaries[result.year]) {
                finalMonthlySummaries[result.year] = {};
                monthlyForSalesforce[result.year] = {};
            }
            finalMonthlySummaries[result.year][result.month] = result;
            monthlyForSalesforce[result.year][result.month] = {
                summary: JSON.stringify(result.aiOutput),
                summaryDetails: result.aiOutput?.summary || '',
                count: result.count,
                startdate: result.startdate
            };
        });

        if (Object.keys(monthlyForSalesforce).length > 0) {
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap, loggedinUserId);
        }

        // 4. Group by Quarter and Prepare for PARALLEL Quarterly Generation
        const quarterlyInputGroups = {};
        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const monthData = finalMonthlySummaries[year][month];
                const quarter = getQuarterFromMonthIndex(monthData.monthIndex);
                const quarterKey = `${year}-${quarter}`;
                if (!quarterlyInputGroups[quarterKey]) quarterlyInputGroups[quarterKey] = [];
                quarterlyInputGroups[quarterKey].push(monthData.aiOutput);
            }
        }
        console.log(`[${accountId}] Identified ${Object.keys(quarterlyInputGroups).length} quarters for parallel generation.`);

        // 5. --- PARALLEL: Generate Quarterly Summaries ---
        const quarterlyPromises = Object.entries(quarterlyInputGroups).map(([quarterKey, monthlySummaries]) => {
            const [year, quarter] = quarterKey.split('-');
            const quarterlyInputDataString = JSON.stringify(monthlySummaries, null, 2);
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate...\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            return generateSummary(
                null, openai, finalQuarterlyAssistantId, userPromptQuarterly, finalQuarterlyFuncSchema
            ).then(aiOutput => ({ quarterKey, aiOutput }))
            .catch(error => {
                console.error(`[${accountId}] Error generating summary for ${quarterKey}:`, error.message);
                return null;
            });
        });

        const quarterlyResults = (await Promise.all(quarterlyPromises)).filter(Boolean);
        console.log(`[${accountId}] Parallel quarterly generation complete. ${quarterlyResults.length} summaries created.`);

        // 6. Transform and Save Quarterly Summaries
        const finalQuarterlyDataForSalesforce = {};
        quarterlyResults.forEach(({ quarterKey, aiOutput }) => {
            const transformedResult = transformQuarterlyStructure(aiOutput);
            for (const year in transformedResult) {
                if (!finalQuarterlyDataForSalesforce[year]) finalQuarterlyDataForSalesforce[year] = {};
                Object.assign(finalQuarterlyDataForSalesforce[year], transformedResult[year]);
            }
        });

        if (Object.keys(finalQuarterlyDataForSalesforce).length > 0) {
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId);
        }

        // 7. Send Success Callback
        if(sendCallback == 'Yes') {
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", "Summary Processed Successfully");
        }
    } catch (error) {
        console.error(`[${accountId}] Error during async summary processing:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Processing error: ${error.message}`);
    }
}


// --- NEW: Synchronous Summary Generation for Recent Data ---
async function generateRecentSummary(
    accountId, accessToken, userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId, finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId
) {
    console.log(`[${accountId}] Executing synchronous flow.`);
    const conn = new jsforce.Connection({ instanceUrl: SF_LOGIN_URL, accessToken: accessToken, version: '59.0' });

    // 1. Modify SOQL to fetch only recent records (last 60 days to catch current/previous month)
    let recentQuery = queryText;
    const whereRegex = /\sWHERE\s/i;
    const dateFilter = "CreatedDate = LAST_N_DAYS:60";

    if (whereRegex.test(recentQuery)) {
        recentQuery = recentQuery.replace(whereRegex, ` WHERE ${dateFilter} AND `);
    } else {
        const fromRegex = /\sFROM\s/i;
        recentQuery = recentQuery.replace(fromRegex, ` WHERE ${dateFilter} FROM `);
    }
    console.log(`[${accountId}] Using modified query for recent data: ${recentQuery}`);

    // 2. Fetch and find the most recent month with data
    const groupedData = await fetchRecords(conn, recentQuery);
    if (Object.keys(groupedData).length === 0) {
        console.log(`[${accountId}] No recent activities found in the last 60 days.`);
        return { message: "No recent activities found to summarize." };
    }

    // Find the latest year, month, and activities
    const latestYear = Math.max(...Object.keys(groupedData).map(Number));
    const latestYearData = groupedData[latestYear];
    // This assumes the last month in the grouped array is the most recent, which should be correct.
    const latestMonthObj = latestYearData[latestYearData.length - 1];
    const latestMonthName = Object.keys(latestMonthObj)[0];
    const latestActivities = latestMonthObj[latestMonthName];
    console.log(`[${accountId}] Identified most recent data for: ${latestMonthName} ${latestYear}`);

    // 3. Generate summary for the most recent month
    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${latestMonthName} ${latestYear}`);
    const monthlyAiOutput = await generateSummary(
        latestActivities, openai, finalMonthlyAssistantId, userPromptMonthly, finalMonthlyFuncSchema
    );
    console.log(`[${accountId}] Generated monthly summary for ${latestMonthName} ${latestYear}.`);

    // 4. Save the monthly summary to Salesforce
    const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
    const monthIndex = monthMap[latestMonthName.toLowerCase()];
    const monthlyForSalesforce = {
        [latestYear]: {
            [latestMonthName]: {
                summary: JSON.stringify(monthlyAiOutput),
                summaryDetails: monthlyAiOutput?.summary || '',
                count: latestActivities.length,
                startdate: new Date(Date.UTC(latestYear, monthIndex, 1)).toISOString().split('T')[0]
            }
        }
    };
    await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap, loggedinUserId);
    console.log(`[${accountId}] Saved monthly summary to Salesforce.`);

    // 5. Generate and save the corresponding quarterly summary
    // To do this properly, we need the *other* monthly summaries for that quarter.
    // For simplicity and speed, we will generate a quarterly summary using only the available monthly data.
    // A more advanced implementation might fetch existing monthly summaries from Salesforce first.
    let quarterlyAiOutput = null;
    const quarter = getQuarterFromMonthIndex(monthIndex);
    const quarterKey = `${latestYear}-${quarter}`;
    const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', latestYear)}\n\nAggregate...\n\`\`\`json\n${JSON.stringify([monthlyAiOutput], null, 2)}\n\`\`\``;

    console.log(`[${accountId}] Generating quarterly summary for ${quarterKey}. NOTE: This will be based on the single month just processed.`);
    quarterlyAiOutput = await generateSummary(
        null, openai, finalQuarterlyAssistantId, userPromptQuarterly, finalQuarterlyFuncSchema
    );
    console.log(`[${accountId}] Generated quarterly summary for ${quarterKey}.`);

    const transformedQuarterly = transformQuarterlyStructure(quarterlyAiOutput);
    if (Object.keys(transformedQuarterly).length > 0) {
        await createTimileSummarySalesforceRecords(conn, transformedQuarterly, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId);
        console.log(`[${accountId}] Saved quarterly summary to Salesforce.`);
    }

    // 6. Return the generated data
    return {
        monthlySummary: monthlyAiOutput,
        quarterlySummary: quarterlyAiOutput
    };
}


// --- OpenAI Summary Generation Function (No changes here) ---
async function generateSummary(
    activities, openaiClient, assistantId, userPrompt, functionSchema
) {
    let fileId = null, thread = null, filePath = null;
    let inputMethod = "prompt";
    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId}`);
        let finalUserPrompt = userPrompt, messageAttachments = [];
        if (activities && Array.isArray(activities) && activities.length > 0) {
            let potentialFullPrompt, activitiesJsonString;
            try {
                 activitiesJsonString = JSON.stringify(activities, null, 2);
                 potentialFullPrompt = `${userPrompt}\n\nHere is the activity data to process:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;
            } catch(stringifyError) {
                throw new Error("Failed to stringify activity data for processing.");
            }
            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                finalUserPrompt = potentialFullPrompt;
            } else {
                inputMethod = "file upload";
                finalUserPrompt = userPrompt;
                let activitiesText = activities.map((activity, index) => {
                    return [`Activity ${index + 1}:`, ...Object.entries(activity).map(([key, value]) => `  ${key}: ${String(value)}`)].join('\n');
                }).join('\n\n---\n\n');
                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                const filename = `salesforce_activities_${timestamp}_${thread.id}.txt`;
                filePath = path.join(TEMP_FILE_DIR, filename);
                await fs.writeFile(filePath, activitiesText);
                const uploadResponse = await openaiClient.files.create({ file: fs.createReadStream(filePath), purpose: "assistants" });
                fileId = uploadResponse.id;
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
            }
        }
        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) messagePayload.attachments = messageAttachments;
        await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added (using ${inputMethod}), starting run...`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }],
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        if (run.status === 'requires_action') {
            const toolCall = run.required_action?.submit_tool_outputs?.tool_calls[0];
            if (!toolCall || toolCall.type !== 'function' || toolCall.function.name !== functionSchema.name) {
                  throw new Error(`Assistant required action for unexpected tool: ${toolCall?.function?.name || toolCall?.type}`);
            }
            const rawArgs = toolCall.function.arguments;
            try {
                 return JSON.parse(rawArgs);
             } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON:`, rawArgs);
                 throw new Error(`Failed to parse function call arguments from AI: ${parseError.message}`);
             }
         } else {
             console.error(`[Thread ${thread.id}] Run ended with unexpected status: ${run.status}`, run.last_error);
             const errorMessage = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : `Run completed without making the required function call to ${functionSchema.name}.`;
             throw new Error(`Assistant run failed. Status: ${run.status}. Error: ${errorMessage}`);
         }
    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`, error);
        throw error;
    } finally {
        if (filePath) { try { await fs.unlink(filePath); } catch (e) { console.error(`Error deleting temp file ${filePath}:`, e); } }
        if (fileId) { try { await openaiClient.files.del(fileId); } catch (e) { if (!(e instanceof NotFoundError)) { console.error(`Error deleting OpenAI file ${fileId}:`, e); } } }
    }
}


// --- Salesforce Record Creation/Update Function (No changes here) ---
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap,loggedinUserId) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} summaries...`);
    let recordsToCreate = [], recordsToUpdate = [];
    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const summaryData = summaries[year][periodKey];
            let summaryJsonString = summaryData.summaryJson || summaryData.summary;
            let summaryDetailsHtml = summaryData.summaryDetails || '';
            if (!summaryDetailsHtml && summaryJsonString) {
                try { summaryDetailsHtml = JSON.parse(summaryJsonString)?.summary || ''; } catch (e) { console.warn(`Could not parse JSON to extract details for ${periodKey} ${year}.`); }
            }
            let fyQuarterValue = (summaryCategory === 'Quarterly') ? periodKey : '';
            let monthValue = (summaryCategory === 'Monthly') ? periodKey : '';
            let shortMonth = monthValue ? monthValue.substring(0, 3) : '';
            let summaryMapKey = (summaryCategory === 'Quarterly') ? `${fyQuarterValue} ${year}` : `${shortMonth} ${year}`;
            let existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);
            const recordPayload = {
                Parent_Id__c: parentId,
                Month__c: monthValue || null,
                Year__c: String(year),
                Summary_Category__c: summaryCategory,
                Requested_By__c: loggedinUserId,
                Type__c: 'Activity',
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null,
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null,
                FY_Quarter__c: fyQuarterValue || null,
                Month_Date__c: summaryData.startdate,
                Number_of_Records__c: summaryData.count || 0,
            };
            if (!recordPayload.Summary_Category__c || !recordPayload.Month_Date__c) {
                 console.warn(`[${parentId}] Skipping record for ${summaryMapKey} due to missing data.`);
                 continue;
            }
            if (existingRecordId) {
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                recordsToCreate.push(recordPayload);
            }
        }
    }
    try {
        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} summary records...`);
            await conn.sobject(TIMELINE_SUMMARY_OBJECT_API_NAME).create(recordsToCreate, { allOrNone: false });
        }
        if (recordsToUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${recordsToUpdate.length} existing ${summaryCategory} summary records...`);
            await conn.sobject(TIMELINE_SUMMARY_OBJECT_API_NAME).update(recordsToUpdate, { allOrNone: false });
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} records to Salesforce: ${err.message}`, err);
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}


// --- Salesforce Data Fetching with Pagination (No changes here) ---
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        const queryResult = isFirstIteration ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);
        if (queryResult.records && queryResult.records.length > 0) {
            allRecords.push(...queryResult.records);
        }
        if (!queryResult.done && queryResult.nextRecordsUrl) {
            await new Promise(resolve => setTimeout(resolve, 200));
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        } else {
            return groupRecordsByMonthYear(allRecords);
        }
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce activities: ${error.message}`, error);
        throw error;
    }
}


// --- Data Grouping Helper Function (No changes here) ---
function groupRecordsByMonthYear(records) {
    const groupedData = {};
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    records.forEach(activity => {
        if (!activity.CreatedDate) return;
        try {
            const date = new Date(activity.CreatedDate);
            if (isNaN(date.getTime())) return;
            const year = date.getUTCFullYear();
            const monthIndex = date.getUTCMonth();
            const month = monthNames[monthIndex];
            if (!groupedData[year]) groupedData[year] = [];
            let monthEntry = groupedData[year].find(entry => entry[month]);
            if (!monthEntry) {
                monthEntry = { [month]: [] };
                groupedData[year].push(monthEntry);
            }
            monthEntry[month].push({
                Id: activity.Id,
                Description: activity.Description || null,
                Subject: activity.Subject || null,
                CreatedDate: activity.CreatedDate
            });
        } catch(dateError) {
             console.warn(`Skipping activity (ID: ${activity.Id}) due to date processing error: ${dateError.message}`);
        }
    });
    // Ensure months within a year are sorted chronologically
    for (const year in groupedData) {
        groupedData[year].sort((a, b) => {
            const monthA = Object.keys(a)[0];
            const monthB = Object.keys(b)[0];
            return monthNames.indexOf(monthA) - monthNames.indexOf(monthB);
        });
    }
    return groupedData;
}


// --- Callback Sending Function (No changes here) ---
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    console.log(`[${accountId}] Sending ASYNC callback to ${callbackUrl}. Status: ${status}`);
    try {
        await axios.post(callbackUrl, {
            accountId: accountId,
            loggedinUserId: loggedinUserId,
            status: "Completed",
            processResult: (status === "Success" || status === "Failed") ? status : "Failed",
            message: message
        }, {
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${accessToken}`
            },
            timeout: 30000
        });
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        console.error(`[${accountId}] Failed to send callback to ${callbackUrl}:`, error.message);
    }
}


// --- Utility Helper Functions (No changes here) ---
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && typeof item.key === 'string' && item.key.toLowerCase() === searchKey.toLowerCase());
    return record ? record.value : null;
}

function transformQuarterlyStructure(quarterlyAiOutput) {
    const result = {};
    if (!quarterlyAiOutput?.yearlySummary?.[0]?.quarters?.[0]) {
        console.warn("Invalid structure received from quarterly AI for transformation:", JSON.stringify(quarterlyAiOutput));
        return result;
    }
    const yearData = quarterlyAiOutput.yearlySummary[0];
    const year = yearData.year;
    const quarterData = yearData.quarters[0];
    const quarter = quarterData.quarter;
    if (!year || !quarter || !quarter.match(/^Q[1-4]$/i)) {
        console.warn(`Invalid year/quarter in AI output: Year=${year}, Quarter=${quarter}`);
        return result;
    }
    let startDate = quarterData.startdate;
    if (!startDate || !/^\d{4}-\d{2}-\d{2}$/.test(startDate)) {
        startDate = `${year}-${getQuarterStartMonth(quarter)}-01`;
    }
    if (!result[year]) result[year] = {};
    result[year][quarter] = {
        summaryDetails: quarterData.summary || '',
        summaryJson: JSON.stringify(quarterData),
        count: quarterData.activityCount || 0,
        startdate: startDate
    };
    return result;
}

function getQuarterStartMonth(quarter) {
    switch (quarter.toUpperCase()) {
        case 'Q1': return '01';
        case 'Q2': return '04';
        case 'Q3': '07';
        case 'Q4': return '10';
        default: return '01';
    }
}
