/*
 * Enhanced Node.js Express application for generating Salesforce activity and CTA summaries using OpenAI Assistants.
 *
 * Features:
 * - Creates/Retrieves TWO OpenAI Assistants (Monthly & Quarterly) on startup using environment variable IDs.
 * - Asynchronous processing with immediate acknowledgement.
 * - Salesforce integration (fetching activities/CTAs, saving summaries).
 * - OpenAI Assistants API V2 usage.
 * - Dynamic function schema loading (default or from request, specific to Activity/CTA).
 * - Dynamic tool_choice to force specific function calls.
 * - Conditional input method: Direct JSON in prompt (< threshold) or File Upload (>= threshold).
 * - Generates summaries per month and aggregates per relevant quarter individually for Activities or CTAs.
 * - Robust error handling and callback mechanism.
 * - Temporary file management.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables ---
dotenv.config();

// --- Configuration Constants ---
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const OPENAI_MONTHLY_ASSISTANT_ID_ENV = process.env.OPENAI_MONTHLY_ASSISTANT_ID;
const OPENAI_QUARTERLY_ASSISTANT_ID_ENV = process.env.OPENAI_QUARTERLY_ASSISTANT_ID;

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c";
const DIRECT_INPUT_THRESHOLD = 2000;
const PROMPT_LENGTH_THRESHOLD = 256000;
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files');

if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

let monthlyAssistantId = null;
let quarterlyAssistantId = null;

// --- Default OpenAI Function Schemas (Activity) ---
const defaultActivityFunctions = [
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
                          "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
                          "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                        },
                        "required": ["Id", "LinkText", "ActivityDate"],
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
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
                            "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "ActivityDate"],
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
                            "LinkText": { "type": "string", "description": "'MMM DD YYYY: Short Description (max 50 chars)' - Generate from ActivityDate, Subject, Description." },
                            "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format. Must belong to the summarized month/year." }
                          },
                          "required": ["Id", "LinkText", "ActivityDate"],
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
                                  "ActivityDate": { "type": "string", "description": "Activity Date in 'YYYY-MM-DD' format (copied from monthly input)." }
                                },
                                "required": ["id", "linkText", "ActivityDate"],
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

// --- Default OpenAI Function Schemas (CTA) ---
const defaultCtaFunctions = [
    {
        "name": "generate_monthly_cta_summary",
        "description": "Generates a structured monthly CTA insight summary and an HTML report string based on provided CTA data.",
        "parameters": {
            "type": "object",
            "properties": {
                "month": { "type": "string", "description": "Month and year of the summary, e.g., 'January 2025'" },
                "total_ctas": { "type": "integer" },
                "converted_ctas": { "type": "integer" },
                "rejected_ctas": { "type": "integer" },
                "sla_analysis": {
                    "type": "object", "properties": { "converted_avg_sla": { "type": "number" }, "rejected_avg_sla": { "type": "number" }, "action": { "type": "string" } },
                    "required": ["converted_avg_sla", "rejected_avg_sla", "action"]
                },
                "product_insights": {
                    "type": "object", "properties": { "converted_top_products": { "type": "array", "items": { "type": "string" } }, "rejected_top_products": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } },
                    "required": ["converted_top_products", "rejected_top_products", "action"]
                },
                "score_grade_insights": {
                    "type": "object", "properties": { "high_score_stats": { "type": "string" }, "low_score_stats": { "type": "string" }, "action": { "type": "string" } },
                    "required": ["high_score_stats", "low_score_stats", "action"]
                },
                "conversion_vs_rejection_metrics": {
                    "type": "object",
                    "properties": {
                        "avg_sla_days": { "type": "object", "properties": { "converted": { "type": "number" }, "rejected": { "type": "number" } } },
                        "most_common_product": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } },
                        "top_score_grade": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } },
                        "fastest_followup_hrs": { "type": "object", "properties": { "converted": { "type": "integer" }, "rejected": { "type": "integer" } } },
                        "common_rejected_reason": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } }
                    },
                    "required": ["avg_sla_days", "most_common_product", "top_score_grade", "fastest_followup_hrs", "common_rejected_reason"]
                },
                "score_and_product_breakdown": {
                    "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } },
                    "required": ["converted", "rejected"]
                },
                "common_rejected_reasons": {
                    "type": "object", "properties": { "top_reasons": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } },
                    "required": ["top_reasons", "action"]
                },
                "description_insights": {
                    "type": "object", "properties": { "topics": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } },
                    "required": ["topics", "action"]
                },
                "action_list": { "type": "array", "items": { "type": "string" } },
                "html_report": {
                    "type": "string",
                    "description": "The complete HTML summary report for the month, generated based on the other parameters."
                }
            },
            "required": [
                "month", "total_ctas", "converted_ctas", "rejected_ctas",
                "sla_analysis", "product_insights", "score_grade_insights",
                "conversion_vs_rejection_metrics", "score_and_product_breakdown",
                "common_rejected_reasons", "description_insights", "action_list",
                "html_report"
            ]
        }
    },
    {
        "name": "generate_quarterly_cta_summary",
        "description": "Aggregates multiple monthly structured CTA summaries into one quarterly structured CTA summary and an HTML report string.",
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": { "type": "string", "description": "Quarter and year, e.g., 'Q1 2025'" },
                "monthly_summaries": {
                    "type": "array",
                    "description": "Array of structured monthly CTA summary inputs (output from 'generate_monthly_cta_summary' calls, excluding their 'html_report' field).",
                    "items": { "$ref": "#/definitions/monthly_summary_cta_input_for_quarterly" }
                },
                "aggregated_data": {
                    "type": "object",
                    "description": "Aggregated data for the quarter, similar in structure to a single month's data but summarized across the quarter.",
                    "properties": {
                        "total_ctas": { "type": "integer" }, "converted_ctas": { "type": "integer" }, "rejected_ctas": { "type": "integer" },
                        "sla_analysis": { "$ref": "#/definitions/cta_sla_analysis" },
                        "product_insights": { "$ref": "#/definitions/cta_product_insights" },
                        "score_grade_insights": { "$ref": "#/definitions/cta_score_grade_insights" },
                        "conversion_vs_rejection_metrics": { "$ref": "#/definitions/cta_conversion_vs_rejection_metrics" },
                        "score_and_product_breakdown": { "$ref": "#/definitions/cta_score_and_product_breakdown" },
                        "common_rejected_reasons": { "$ref": "#/definitions/cta_common_rejected_reasons" },
                        "description_insights": { "$ref": "#/definitions/cta_description_insights" },
                        "action_list": { "type": "array", "items": { "type": "string" } }
                    },
                     "required": [
                        "total_ctas", "converted_ctas", "rejected_ctas",
                        "sla_analysis", "product_insights", "score_grade_insights",
                        "conversion_vs_rejection_metrics", "score_and_product_breakdown",
                        "common_rejected_reasons", "description_insights", "action_list"
                    ]
                },
                "html_report": {
                    "type": "string",
                    "description": "The complete HTML summary report for the quarter, generated based on the aggregated data."
                }
            },
            "required": ["quarter", "monthly_summaries", "aggregated_data", "html_report"],
            "definitions": {
                "monthly_summary_cta_input_for_quarterly": {
                    "type": "object",
                    "properties": {
                        "month": { "type": "string" }, "total_ctas": { "type": "integer" }, "converted_ctas": { "type": "integer" }, "rejected_ctas": { "type": "integer" },
                        "sla_analysis": { "$ref": "#/definitions/cta_sla_analysis" },
                        "product_insights": { "$ref": "#/definitions/cta_product_insights" },
                        "score_grade_insights": { "$ref": "#/definitions/cta_score_grade_insights" },
                        "conversion_vs_rejection_metrics": { "$ref": "#/definitions/cta_conversion_vs_rejection_metrics" },
                        "score_and_product_breakdown": { "$ref": "#/definitions/cta_score_and_product_breakdown" },
                        "common_rejected_reasons": { "$ref": "#/definitions/cta_common_rejected_reasons" },
                        "description_insights": { "$ref": "#/definitions/cta_description_insights" },
                        "action_list": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": [
                        "month", "total_ctas", "converted_ctas", "rejected_ctas", "sla_analysis", "product_insights",
                        "score_grade_insights", "conversion_vs_rejection_metrics", "score_and_product_breakdown",
                        "common_rejected_reasons", "description_insights", "action_list"
                    ]
                },
                "cta_sla_analysis": { "type": "object", "properties": { "converted_avg_sla": { "type": "number" }, "rejected_avg_sla": { "type": "number" }, "action": { "type": "string" } } },
                "cta_product_insights": { "type": "object", "properties": { "converted_top_products": { "type": "array", "items": { "type": "string" } }, "rejected_top_products": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } } },
                "cta_score_grade_insights": { "type": "object", "properties": { "high_score_stats": { "type": "string" }, "low_score_stats": { "type": "string" }, "action": { "type": "string" } } },
                "cta_conversion_vs_rejection_metrics": {
                    "type": "object", "properties": {
                        "avg_sla_days": { "type": "object", "properties": { "converted": { "type": "number" }, "rejected": { "type": "number" } } },
                        "most_common_product": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } },
                        "top_score_grade": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } },
                        "fastest_followup_hrs": { "type": "object", "properties": { "converted": { "type": "integer" }, "rejected": { "type": "integer" } } },
                        "common_rejected_reason": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } }
                    }
                },
                "cta_score_and_product_breakdown": { "type": "object", "properties": { "converted": { "type": "string" }, "rejected": { "type": "string" } } },
                "cta_common_rejected_reasons": { "type": "object", "properties": { "top_reasons": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } } },
                "cta_description_insights": { "type": "object", "properties": { "topics": { "type": "array", "items": { "type": "string" } }, "action": { "type": "string" } } }
            }
        }
    }
];

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

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
        const role = assistantName.includes("Monthly") ? "MONTHLY" : assistantName.includes("Quarterly") ? "QUARTERLY" : "GENERIC";
        const envVarNameSuggestion = `OPENAI_${role}_ASSISTANT_ID`;
        console.warn(`--> IMPORTANT: Consider adding this ID to your .env file as ${envVarNameSuggestion}=${newAssistant.id} for future reuse if this assistant is intended for this role.`);
        return newAssistant.id;
    } catch (creationError) {
        console.error(`Error creating Assistant "${assistantName}":`, creationError);
        throw new Error(`Failed to create Assistant ${assistantName}: ${creationError.message}`);
    }
}

(async () => {
    try {
        console.log("Initializing Assistants...");
        await fs.ensureDir(TEMP_FILE_DIR);
        const assistantBaseTools = [{ type: "file_search" }, { type: "function" }];

        monthlyAssistantId = await createOrRetrieveAssistant(
            openai, OPENAI_MONTHLY_ASSISTANT_ID_ENV,
            "Salesforce Monthly Data Summarizer",
            "You are an AI assistant. Your task is to analyze Salesforce data (which could be Activities or CTAs) for a single month. You will be provided with the data and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function. If the schema includes an 'html_report' field, you must generate the HTML report content for it. If data is provided as a file, use the file_search tool to access and process its content.",
            assistantBaseTools, OPENAI_MODEL
        );

        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai, OPENAI_QUARTERLY_ASSISTANT_ID_ENV,
            "Salesforce Quarterly Data Aggregator",
            "You are an AI assistant. Your task is to aggregate pre-summarized monthly Salesforce data (which could be from Activities or CTAs) into a consolidated quarterly summary. You will be provided with an array of monthly JSON summaries and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function. This includes correctly processing the input 'monthly_summaries' array and generating aggregated data. If the schema includes an 'html_report' field, you must generate the HTML report content for it.",
            assistantBaseTools, OPENAI_MODEL
        );

        if (!monthlyAssistantId || !quarterlyAssistantId ) {
             throw new Error("Failed to obtain valid IDs for one or more core Assistants during startup.");
        }

        app.listen(PORT, () => {
            console.log("----------------------------------------------------");
            console.log(`Server running on port ${PORT}`);
            console.log(`Using OpenAI Model (for new Assistants): ${OPENAI_MODEL}`);
            console.log(`Using Generic Monthly Assistant ID: ${monthlyAssistantId}`);
            console.log(`Using Generic Quarterly Assistant ID: ${quarterlyAssistantId}`);
            console.log(`Direct JSON input threshold: ${DIRECT_INPUT_THRESHOLD} records`);
            console.log(`Prompt length threshold for file upload: ${PROMPT_LENGTH_THRESHOLD} characters`);
            console.log(`Temporary file directory: ${TEMP_FILE_DIR}`);
            console.log("----------------------------------------------------");
        });

    } catch (startupError) {
        console.error("FATAL STARTUP ERROR:", startupError.message);
        process.exit(1);
    }
})();

app.post('/generatesummary', async (req, res) => {
    console.log("Received /generatesummary request");
    const {
        accountId, callbackUrl, userPrompt, userPromptQtr, queryText,
        summaryMap, loggedinUserId, qtrJSON, monthJSON,
        summarObj
    } = req.body;

    if (!monthlyAssistantId || !quarterlyAssistantId) {
        console.error("Error: Core Assistants not initialized properly.");
        return res.status(500).json({ error: "Internal Server Error: Core Assistants not ready." });
    }
    if (summarObj !== "Activity" && summarObj !== "CTA") {
        return res.status(400).send({ error: "Invalid 'summarObj' parameter. Must be 'Activity' or 'CTA'." });
    }

    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ error: "Unauthorized" });
    }
    const accessToken = authHeader.split(" ")[1];

    if (!accountId || !callbackUrl || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId || !summarObj) {
        return res.status(400).send({ error: "Missing required parameters" });
    }

    let summaryRecordsMap = {};
    let finalMonthlyFuncSchema, finalQuarterlyFuncSchema;
    const currentMonthlyAssistantId = monthlyAssistantId;
    const currentQuarterlyAssistantId = quarterlyAssistantId;

    if (summarObj === "Activity") {
        finalMonthlyFuncSchema = defaultActivityFunctions.find(f => f.name === 'generate_monthly_activity_summary');
        finalQuarterlyFuncSchema = defaultActivityFunctions.find(f => f.name === 'generate_quarterly_activity_summary');
    } else if (summarObj === "CTA") {
        finalMonthlyFuncSchema = defaultCtaFunctions.find(f => f.name === 'generate_monthly_cta_summary');
        finalQuarterlyFuncSchema = defaultCtaFunctions.find(f => f.name === 'generate_quarterly_cta_summary');
    }

    try {
        if (summaryMap) summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([k, v]) => ({ key:k, value:v }));
        if (monthJSON) {
            const customMonthSchema = JSON.parse(monthJSON);
            if (!customMonthSchema?.name || !customMonthSchema?.parameters) throw new Error("MonthJSON invalid.");
            finalMonthlyFuncSchema = customMonthSchema;
            console.log(`Using custom monthly schema for ${summarObj}.`);
        }
        if (qtrJSON) {
            const customQtrSchema = JSON.parse(qtrJSON);
            if (!customQtrSchema?.name || !customQtrSchema?.parameters) throw new Error("QtrJSON invalid.");
            finalQuarterlyFuncSchema = customQtrSchema;
            console.log(`Using custom quarterly schema for ${summarObj}.`);
        }
    } catch (e) {
        return res.status(400).send({ error: `Invalid JSON. ${e.message}` });
    }

    if (!finalMonthlyFuncSchema || !finalQuarterlyFuncSchema) {
        return res.status(500).send({ error: `Could not load schemas for ${summarObj}.`});
    }

    res.status(202).json({ status: 'processing', message: `Summary for ${summarObj} initiated.` });
    console.log(`Initiating ${summarObj} summary for Account: ${accountId} MonthlyAsst: ${currentMonthlyAssistantId}, QtrlyAsst: ${currentQuarterlyAssistantId}`);

    processSummary(
        accountId, accessToken, callbackUrl, userPrompt, userPromptQtr, queryText,
        summaryRecordsMap, loggedinUserId,
        finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
        currentMonthlyAssistantId, currentQuarterlyAssistantId,
        summarObj
    ).catch(async (error) => {
        console.error(`[${accountId}] Unhandled error for ${summarObj}:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled ${summarObj} error: ${error.message}`);
    });
});

function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex <= 2) return 'Q1'; if (monthIndex <= 5) return 'Q2';
    if (monthIndex <= 8) return 'Q3'; if (monthIndex <= 11) return 'Q4';
    return 'Unknown';
}

async function processSummary(
    accountId, accessToken, callbackUrl,
    userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId,
    finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId,
    summarObj
) {
    console.log(`[${accountId}] processSummary for ${summarObj} MonthlyAsst: ${finalMonthlyAssistantId}, QtrlyAsst: ${finalQuarterlyAssistantId}`);
    const conn = new jsforce.Connection({ instanceUrl: SF_LOGIN_URL, accessToken, maxRequest: 5, version: '59.0' });

    try {
        const groupedData = await fetchRecords(conn, queryText, summarObj);
        const recordCount = Object.values(groupedData).flat().reduce((sum, monthObj) => sum + Object.values(monthObj)[0].length, 0);
        console.log(`[${accountId}] Fetched/grouped ${summarObj} data. Records: ${recordCount}`);

        const finalMonthlySummaries = {};
        const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };

        for (const year in groupedData) {
            finalMonthlySummaries[year] = {};
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    const records = monthObj[month];
                    if (records.length === 0) continue;
                    const monthIndex = monthMap[month.toLowerCase()];
                    if (monthIndex === undefined) continue;

                    const startDate = new Date(Date.UTC(year, monthIndex, 1));
                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);
                    const monthlySummaryResult = await generateSummary(
                        records, openai, finalMonthlyAssistantId,
                        userPromptMonthly, finalMonthlyFuncSchema, summarObj
                    );
                    finalMonthlySummaries[year][month] = {
                         aiOutput: monthlySummaryResult, count: records.length,
                         startdate: startDate.toISOString().split('T')[0],
                         year: parseInt(year), monthIndex
                    };
                    console.log(`[${accountId}] Monthly ${summarObj} summary for ${month} ${year} done.`);
                 }
            }
        }

        const monthlyForSalesforce = {};
        for (const year in finalMonthlySummaries) {
             monthlyForSalesforce[year] = {};
             for (const month in finalMonthlySummaries[year]) {
                 const mData = finalMonthlySummaries[year][month];
                 let html = (summarObj === "Activity") ? mData.aiOutput?.summary : mData.aiOutput?.html_report;
                 monthlyForSalesforce[year][month] = {
                     summary: JSON.stringify(mData.aiOutput), summaryDetails: html || '',
                     count: mData.count, startdate: mData.startdate
                 };
             }
        }
        if (Object.values(monthlyForSalesforce).some(y => Object.keys(y).length > 0)) {
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap, loggedinUserId, summarObj);
        }

        const quarterlyInputGroups = {};
        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const mData = finalMonthlySummaries[year][month];
                const qKey = `${year}-${getQuarterFromMonthIndex(mData.monthIndex)}`;
                if (!quarterlyInputGroups[qKey]) quarterlyInputGroups[qKey] = [];
                let aiOut = mData.aiOutput;
                if (summarObj === "CTA" && aiOut) { const { html_report, ...rest } = aiOut; aiOut = rest; }
                quarterlyInputGroups[qKey].push(aiOut);
            }
        }

        const allQuarterlyRawResults = {};
        for (const [qKey, monthSummaries] of Object.entries(quarterlyInputGroups)) {
            if (!monthSummaries || monthSummaries.length === 0) continue;
            const [year, qtr] = qKey.split('-');
            const promptQtr = `${userPromptQuarterlyTemplate.replace('{{Quarter}}',qtr).replace('{{Year}}',year)}\n\nAgg ${summarObj} data for ${qKey}:\n\`\`\`json\n${JSON.stringify(monthSummaries,null,2)}\n\`\`\``;
            try {
                 allQuarterlyRawResults[qKey] = await generateSummary(
                    null, openai, finalQuarterlyAssistantId, promptQtr, finalQuarterlyFuncSchema, summarObj
                 );
                 console.log(`[${accountId}] Quarterly ${summarObj} summary for ${qKey} done.`);
            } catch (qErr) { console.error(`[${accountId}] Fail Qtrly ${summarObj} for ${qKey}:`, qErr); }
        }

        const finalQuarterlyDataForSalesforce = {};
        for (const [qKey, rawAi] of Object.entries(allQuarterlyRawResults)) {
             const transformed = transformQuarterlyStructure(rawAi, qKey, summarObj);
             for (const year in transformed) {
                 if (!finalQuarterlyDataForSalesforce[year]) finalQuarterlyDataForSalesforce[year] = {};
                 for (const qtr in transformed[year]) {
                     finalQuarterlyDataForSalesforce[year][qtr] = transformed[year][qtr];
                 }
             }
        }
        if (Object.values(finalQuarterlyDataForSalesforce).some(y => Object.keys(y).length > 0)) {
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId, summarObj);
        }

        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", `${summarObj} Processed`);
    } catch (error) {
        console.error(`[${accountId}] Error in ${summarObj} processSummary:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `${summarObj} Error: ${error.message}`);
    }
}

async function generateSummary(
    records, openaiClient, assistantId, userPrompt, functionSchema, summarObj
) {
    let fileId = null, thread = null, filePath = null, inputMethod = "prompt";
    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        thread = await openaiClient.beta.threads.create();
        console.log(`[Th ${thread.id}] Asst ${assistantId} (${summarObj}, Func: ${functionSchema.name})`);

        let finalUserPrompt = userPrompt, attachments = [];
        if (records?.length > 0) {
            let recJson = JSON.stringify(records, null, 2);
            let potentialPrompt = `${userPrompt}\n\n${summarObj} data:\n\`\`\`json\n${recJson}\n\`\`\``;
            if (potentialPrompt.length < PROMPT_LENGTH_THRESHOLD && records.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON"; finalUserPrompt = potentialPrompt;
            } else {
                inputMethod = "file upload";
                let recText = records.map((r, i) => `${summarObj} ${i+1}:\n`+Object.entries(r).map(([k,v])=>`  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join('\n')).join('\n\n---\n\n');
                filePath = path.join(TEMP_FILE_DIR, `sf_${summarObj.toLowerCase()}_${Date.now()}_${thread.id}.txt`);
                await fs.writeFile(filePath, recText);
                const upRes = await openaiClient.files.create({ file: fs.createReadStream(filePath), purpose: "assistants" });
                fileId = upRes.id;
                attachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
            }
        }
        console.log(`[Th ${thread.id}] Using ${inputMethod} for ${summarObj}.`);

        const msgPayload = { role: "user", content: finalUserPrompt };
        if (attachments.length > 0) msgPayload.attachments = attachments;
        await openaiClient.beta.threads.messages.create(thread.id, msgPayload);

        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }],
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Th ${thread.id}] Run status ${summarObj}: ${run.status}`);

        if (run.status === 'requires_action') {
            const toolCall = run.required_action?.submit_tool_outputs?.tool_calls?.[0];
            if (!toolCall || toolCall.type !== 'function' || toolCall.function.name !== functionSchema.name) {
                  throw new Error(`Unexpected tool: Expected ${functionSchema.name}, Got ${toolCall?.function?.name || toolCall?.type}`);
            }
            console.log(`[Th ${thread.id}] Args for ${summarObj} (${toolCall.function.name}) recvd.`);
            return JSON.parse(toolCall.function.arguments);
        } else if (run.status === 'completed') {
             const msgs = await openaiClient.beta.threads.messages.list(run.thread_id, {limit:1});
             throw new Error(`${summarObj} run completed, no func call. Last msg: ${msgs.data[0]?.content[0]?.text?.value || "N/A"}`);
        } else {
             throw new Error(`${summarObj} run failed. Status: ${run.status}. Err: ${run.last_error?.message || 'Unknown'}`);
        }
    } catch (error) {
        console.error(`[Th ${thread?.id || 'N/A'}] generateSummary err (${summarObj}): ${error.message}`, error);
        throw error;
    } finally {
        if (filePath) try { await fs.unlink(filePath); } catch (e) { console.error(`Err del temp ${filePath}:`,e); }
        if (fileId) try { await openaiClient.files.del(fileId); } catch (e) { if (!(e instanceof NotFoundError || e?.status === 404)) console.error(`Err del OpenAI file ${fileId}:`,e); }
    }
}

async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap, loggedinUserId, summarObj) {
    console.log(`[${parentId}] Saving ${summaryCategory} ${summarObj} summaries...`);
    let toCreate = [], toUpdate = [];
    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const sData = summaries[year][periodKey];
            let jsonStr = sData.summary, html = sData.summaryDetails, startDate = sData.startdate, count = sData.count;
            if (!html && jsonStr) {
                try {
                    const pJson = JSON.parse(jsonStr);
                    html = (summarObj === "Activity") ? pJson?.summary : pJson?.html_report;
                } catch (e) { /* ignore */ }
            }
            let qtrVal = (summaryCategory === 'Quarterly') ? periodKey : null;
            let monthVal = (summaryCategory === 'Monthly') ? periodKey : null;
            let mapKey = (summaryCategory === 'Quarterly') ? `${qtrVal} ${year}` : `${monthVal.substring(0,3)} ${year}`;
            let existId = getValueByKey(summaryRecordsMap, mapKey);

            const payload = {
                Parent_Id__c: parentId, Month__c: monthVal, Year__c: String(year), Summary_Category__c: summaryCategory,
                Requested_By__c: loggedinUserId, Summary__c: jsonStr?.substring(0, 131072),
                Summary_Details__c: html?.substring(0, 131072), FY_Quarter__c: qtrVal,
                Month_Date__c: startDate, Number_of_Records__c: count || 0, Type__c: summarObj,
            };
            if (!payload.Month_Date__c) continue;
            if (existId) toUpdate.push({ Id: existId, ...payload }); else toCreate.push(payload);
        }
    }
    const opts = { allOrNone: false };
    if (toCreate.length > 0) {
        const res = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", opts, toCreate);
        handleBulkResults(res, toCreate, 'create', parentId, summarObj);
    }
    if (toUpdate.length > 0) {
        const res = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", opts, toUpdate);
        handleBulkResults(res, toUpdate, 'update', parentId, summarObj);
    }
}

function handleBulkResults(results, payloads, opType, parentId, summarObj) {
    let s=0, f=0;
    results.forEach((res, i) => {
        const p = payloads[i];
        const idLog = p.Id || `${p.Month__c || p.FY_Quarter__c} ${p.Year__c}`;
        if (!res.success) { f++; console.error(`[${parentId}] Err ${opType} ${summarObj} ${idLog}: ${JSON.stringify(res.errors)}`); }
        else s++;
    });
    console.log(`[${parentId}] Bulk ${opType} ${summarObj}: ${s} ok, ${f} fail.`);
}

async function fetchRecords(conn, queryOrUrl, summarObj, allRecords = [], first = true) {
    try {
        const logPfx = first ? `Init Query ${summarObj}` : `Next batch ${summarObj}`;
        console.log(`[SF Fetch] ${logPfx}: ${String(queryOrUrl).substring(0,150)}...`);
        const res = first ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);
        const fetched = res.records?.length || 0;
        if (fetched > 0) allRecords.push(...res.records);
        console.log(`[SF Fetch] Got ${fetched} ${summarObj}. Total: ${allRecords.length}. Done: ${res.done}`);
        if (!res.done && res.nextRecordsUrl) {
            await new Promise(r => setTimeout(r, 200));
            return fetchRecords(conn, res.nextRecordsUrl, summarObj, allRecords, false);
        }
        return groupRecordsByMonthYear(allRecords, summarObj);
    } catch (error) {
        console.error(`[SF Fetch] Error ${summarObj}: ${error.message}`, error);
        throw error;
    }
}

function groupRecordsByMonthYear(records, summarObj) {
    const grouped = {};
    const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    records.forEach(r => {
        if (!r.CreatedDate) return;
        try {
            const d = new Date(r.CreatedDate); if (isNaN(d.getTime())) return;
            const y = d.getUTCFullYear(), mIdx = d.getUTCMonth(), mName = months[mIdx];
            if (!grouped[y]) grouped[y] = [];
            let mEntry = grouped[y].find(e => e[mName]);
            if (!mEntry) { mEntry = { [mName]: [] }; grouped[y].push(mEntry); }
            if (summarObj === "Activity") {
                mEntry[mName].push({ Id:r.Id, Description:r.Description, Subject:r.Subject, ActivityDate:r.CreatedDate });
            } else if (summarObj === "CTA") {
                mEntry[mName].push({
                    Id:r.Id, Name:r.Name, CampaignGroup:r.Campaign_Group__c, CampaignSubType:r.Campaign_Sub_Type__c,
                    CampaignType:r.Campaign_Type__c, Contact:r.Contact__c, CreatedDate:r.CreatedDate,
                    CreatedDateCustom:r.Created_Date_Custom__c, CurrentInterestedProductScoreGrade:r.Current_Interested_Product_Score_Grade__c,
                    CustomerSelectedProduct:r.Customer_Selected_Product__c, CustomersPerceivedSLA:r.CustomersPerceivedSLA__c,
                    DateContacted:r.Date_Contacted__c, DateEmailed:r.Date_Emailed__c, Description:r.Description__c,
                    DispositionedDate:r.Dispositioned_Date__c, LeadScoreAccountSecurity:r.Lead_Score_Account_Security__c,
                    LeadScoreContactCenterTwilioFlex:r.Lead_Score_Contact_Center_Twilio_Flex__c,
                    LeadScoreSMSMessaging:r.Lead_Score_SMS_Messaging__c, LeadScoreVoiceIVRSIPTrunking:r.Lead_Score_Voice_IVR_SIP_Trunking__c,
                    MQLStatus:r.MQL_Status__c, MQLType:r.MQL_Type__c, CurrentOwnerFullName:r.Current_Owner_Full_Name__c,
                    Opportunity:r.Opportunity__c, ProductCategory:r.Product_Category__c, ProductType:r.Product_Type__c,
                    QualifiedLeadSource:r.Qualified_Lead_Source__c, RejectedReason:r.Rejected_Reason__c, Title:r.Title__c
                });
            }
        } catch(e) { /* ignore date err */ }
    });
    console.log(`Finished grouping ${summarObj} records.`);
    return grouped;
}

async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logMsg = message.substring(0,500)+(message.length>500?'...':'');
    console.log(`[${accountId}] Callback to ${callbackUrl}. Status: ${status}, Msg: ${logMsg}`);
    try {
        await axios.post(callbackUrl, {
                accountId, loggedinUserId, status: "Completed",
                processResult: (status === "Success" || status === "Failed") ? status : "Failed", message
            }, { headers: { "Content-Type": "application/json", "Authorization": `Bearer ${accessToken}` }, timeout: 30000 }
        );
    } catch (error) {
        let em = `Fail callback ${callbackUrl}. `;
        if (error.response) em += `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}`;
        else if (error.request) em += `No response. ${error.message}`; else em += `Error: ${error.message}`;
        console.error(`[${accountId}] ${em}`);
    }
}

function getValueByKey(arr, key) {
    if (!arr?.length) return null;
    const rec = arr.find(i => i?.key?.toLowerCase() === key.toLowerCase());
    return rec ? rec.value : null;
}

function transformQuarterlyStructure(qAiOutput, qKey, summarObj) {
    const res = {}; const [yStr, qStr] = qKey.split('-'); const year = parseInt(yStr);
    if (!qAiOutput || !year || !qStr) return res;
    let html = '', count = 0, startDate = `${year}-${getQuarterStartMonth(qStr)}-01`;
    if (summarObj === "Activity") {
        const qData = qAiOutput.yearlySummary?.[0]?.quarters?.[0];
        if (qData?.quarter === qStr && qAiOutput.yearlySummary[0].year === year) {
            html = qData.summary; count = qData.activityCount; startDate = qData.startdate || startDate;
        }
    } else if (summarObj === "CTA") {
        html = qAiOutput.html_report; count = qAiOutput.aggregated_data?.total_ctas;
    }
    if (!res[year]) res[year] = {};
    res[year][qStr] = { summaryDetails: html || '', summaryJson: JSON.stringify(qAiOutput), count: count || 0, startdate: startDate };
    return res;
}

function getQuarterStartMonth(q) {
    if (!q) return '01';
    switch (q.toUpperCase()) {
        case 'Q1': return '01'; case 'Q2': return '04';
        case 'Q3': return '07'; case 'Q4': return '10';
        default: return '01';
    }
}
