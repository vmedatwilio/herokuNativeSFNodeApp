/*
 * Enhanced & Optimized Node.js Express application for Salesforce activity summaries.
 *
 * --- KEY IMPROVEMENTS ---
 * - DUAL-MODE PROCESSING:
 *   - "Fast Sync" mode (via `syncForRecentMonths` param) for recent data returns results directly in the response.
 *   - "Async" mode for large historical datasets with a callback mechanism.
 * - PARALLEL PROCESSING:
 *   - All OpenAI API calls for monthly/quarterly summaries are executed concurrently using Promise.allSettled(),
 *     drastically reducing total processing time from minutes to seconds.
 * - EFFICIENT SALESFORCE DML:
 *   - Uses batched DML operations for high-performance record saving.
 * - SELF-CONTAINED & ROBUST:
 *   - Schemas are passed via request body as per requirement.
 *   - Resilient processing handles partial failures without halting the entire job.
 */

// --- Dependencies ---
const express = require('express');
const jsforce = require('jsforce');
const dotenv =require('dotenv');
const { OpenAI, NotFoundError } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");

// --- Load Environment Variables & Configuration ---
dotenv.config();

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

// --- Environment Variable Validation ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Global Variables for Assistant IDs ---
let monthlyAssistantId = null;
let quarterlyAssistantId = null;

// --- OpenAI Client Initialization ---
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// --- Express Application Setup ---
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// --- Helper Function to Create or Retrieve Assistant (Unchanged from original) ---
async function createOrRetrieveAssistant(
    openaiClient, assistantIdEnvVar, assistantName, assistantInstructions, assistantToolsConfig, assistantModel
) {
    if (assistantIdEnvVar) {
        try {
            const retrievedAssistant = await openaiClient.beta.assistants.retrieve(assistantIdEnvVar);
            console.log(`Successfully retrieved existing Assistant "${retrievedAssistant.name}" with ID: ${retrievedAssistant.id}`);
            return retrievedAssistant.id;
        } catch (error) {
            if (error instanceof NotFoundError) {
                console.warn(`Assistant with ID "${assistantIdEnvVar}" not found. Will create a new one for "${assistantName}".`);
            } else {
                throw new Error(`Failed to retrieve Assistant ${assistantName}: ${error.message}`);
            }
        }
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
        console.warn(`--> IMPORTANT: Add this to your .env file as ${envVarName}=${newAssistant.id} for reuse.`);
        return newAssistant.id;
    } catch (creationError) {
        throw new Error(`Failed to create Assistant ${assistantName}: ${creationError.message}`);
    }
}

// --- Server Startup ---
(async () => {
    try {
        console.log("Initializing Assistants...");
        await fs.ensureDir(TEMP_FILE_DIR);

        const assistantBaseTools = [{ type: "file_search" }, { type: "function" }];

        monthlyAssistantId = await createOrRetrieveAssistant(
            openai, OPENAI_MONTHLY_ASSISTANT_ID_ENV, "Salesforce Monthly Summarizer",
            "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Focus on extracting key themes, tone, and recommended actions.",
            assistantBaseTools, OPENAI_MODEL
        );

        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai, OPENAI_QUARTERLY_ASSISTANT_ID_ENV, "Salesforce Quarterly Summarizer",
            "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data into a structured quarterly JSON summary using the provided function 'generate_quarterly_activity_summary'.",
            assistantBaseTools, OPENAI_MODEL
        );

        if (!monthlyAssistantId || !quarterlyAssistantId) {
             throw new Error("Failed to obtain valid IDs for one or more Assistants during startup.");
        }

        app.listen(PORT, () => {
            console.log("----------------------------------------------------");
            console.log(`Server running on port ${PORT}`);
            console.log(`Using Monthly Assistant ID: ${monthlyAssistantId}`);
            console.log(`Using Quarterly Assistant ID: ${quarterlyAssistantId}`);
            console.log("----------------------------------------------------");
        });
    } catch (startupError) {
        console.error("FATAL STARTUP ERROR:", startupError.message);
        process.exit(1);
    }
})();

// --- Main API Endpoint ---
app.post('/generatesummary', async (req, res) => {
    console.log("Received /generatesummary request");

    if (!monthlyAssistantId || !quarterlyAssistantId) {
        return res.status(500).json({ error: "Internal Server Error: Assistants not ready." });
    }

    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ error: "Unauthorized" });
    }
    const accessToken = authHeader.split(" ")[1];

    const {
        accountId, callbackUrl, userPrompt, userPromptQtr, queryText,
        summaryMap, loggedinUserId, sendCallback, qtrJSON, monthJSON,
        syncForRecentMonths // New parameter for "Fast Sync" mode
    } = req.body;

    if (!accountId || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId || !monthJSON || !qtrJSON) {
        return res.status(400).send({ error: "Missing required parameters (accountId, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId, monthJSON, qtrJSON)" });
    }
    // Async mode requires a callbackUrl
    if (!syncForRecentMonths && !callbackUrl) {
         return res.status(400).send({ error: "Missing required parameter `callbackUrl` for async processing." });
    }

    let summaryRecordsMap = {}, monthlyFuncSchema, quarterlyFuncSchema;
    try {
        if (summaryMap) summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        monthlyFuncSchema = JSON.parse(monthJSON);
        quarterlyFuncSchema = JSON.parse(qtrJSON);
        if (!monthlyFuncSchema.name || !quarterlyFuncSchema.name) {
            throw new Error("Provided monthJSON or qtrJSON schema is invalid.");
        }
    } catch (e) {
        return res.status(400).send({ error: `Invalid JSON provided in summaryMap, monthJSON, or qtrJSON. ${e.message}` });
    }

    const commonArgs = {
        accountId, accessToken, userPromptMonthlyTemplate: userPrompt, userPromptQuarterlyTemplate: userPromptQtr,
        queryText, summaryRecordsMap, loggedinUserId, finalMonthlyFuncSchema: monthlyFuncSchema,
        finalQuarterlyFuncSchema: quarterlyFuncSchema, finalMonthlyAssistantId: monthlyAssistantId,
        finalQuarterlyAssistantId: quarterlyAssistantId
    };

    // --- DUAL-MODE LOGIC: Choose between Fast Sync and Background Async ---
    if (syncForRecentMonths && Number(syncForRecentMonths) > 0) {
        // --- FAST SYNC MODE ---
        console.log(`[${accountId}] Starting FAST SYNC mode for last ${syncForRecentMonths} months.`);
        try {
            // Modify query to fetch only recent data
            const dateLiteral = `LAST_N_MONTHS:${Number(syncForRecentMonths)}`;
            const modifiedQuery = queryText.toUpperCase().includes('WHERE')
                ? queryText.replace(/WHERE/i, `WHERE CreatedDate >= ${dateLiteral} AND `)
                : `${queryText} WHERE CreatedDate >= ${dateLiteral}`;
            
            commonArgs.queryText = modifiedQuery;

            const results = await processSummarySync(commonArgs);
            console.log(`[${accountId}] Fast Sync mode completed. Sending results in response.`);
            return res.status(200).json({ status: 'completed', ...results });

        } catch (error) {
            console.error(`[${accountId}] Error during FAST SYNC processing:`, error);
            return res.status(500).json({ status: 'failed', message: `Sync processing error: ${error.message}` });
        }

    } else {
        // --- ASYNC MODE ---
        res.status(202).json({ status: 'processing', message: 'Summary generation initiated. You will receive a callback.' });
        console.log(`[${accountId}] Starting ASYNC mode for all historical data.`);

        processSummaryAsync({
            ...commonArgs,
            callbackUrl,
            sendCallback
        }).catch(async (error) => {
            console.error(`[${accountId}] Unhandled error during ASYNC background processing:`, error);
            try {
                if(sendCallback === 'Yes') {
                    await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled processing error: ${error.message}`);
                }
            } catch (callbackError) {
                console.error(`[${accountId}] Failed to send error callback after unhandled exception:`, callbackError);
            }
        });
    }
});

/**
 * Processes summaries and returns the results directly. Used for "Fast Sync" mode.
 * The core logic is parallelized for maximum speed.
 */
async function processSummarySync({
    accountId, accessToken, userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId, finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId
}) {
    const conn = new jsforce.Connection({ instanceUrl: SF_LOGIN_URL, accessToken: accessToken, maxRequest: 5, version: '59.0' });
    
    // 1. Fetch data
    console.log(`[${accountId}] (Sync) Fetching Salesforce records...`);
    const groupedData = await fetchRecords(conn, queryText);

    // 2. Generate ALL monthly summaries in PARALLEL
    const { finalMonthlySummaries, monthlyGenerationErrors } = await generateAllMonthlySummaries(
        groupedData, accountId, userPromptMonthlyTemplate, finalMonthlyAssistantId, finalMonthlyFuncSchema
    );

    if (monthlyGenerationErrors.length > 0) {
        console.warn(`[${accountId}] (Sync) Encountered ${monthlyGenerationErrors.length} errors during monthly summary generation.`);
        // Decide if this should be a hard failure or just a warning. For now, we continue.
    }

    // 3. Save monthly summaries to SF
    await saveSummariesToSalesforce(conn, finalMonthlySummaries, accountId, 'Monthly', summaryRecordsMap, loggedinUserId);

    // 4. Generate ALL quarterly summaries in PARALLEL
    const { finalQuarterlyDataForSalesforce, quarterlyGenerationErrors } = await generateAllQuarterlySummaries(
        finalMonthlySummaries, accountId, userPromptQuarterlyTemplate, finalQuarterlyAssistantId, finalQuarterlyFuncSchema
    );

     if (quarterlyGenerationErrors.length > 0) {
        console.warn(`[${accountId}] (Sync) Encountered ${quarterlyGenerationErrors.length} errors during quarterly summary generation.`);
    }

    // 5. Save quarterly summaries to SF
    await saveSummariesToSalesforce(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId);

    console.log(`[${accountId}] (Sync) Processing complete.`);
    // Return a structured result for the client
    return {
        monthlySummaries: finalMonthlySummaries,
        quarterlySummaries: finalQuarterlyDataForSalesforce,
        errors: {
            monthly: monthlyGenerationErrors,
            quarterly: quarterlyGenerationErrors
        }
    };
}


/**
 * Processes summaries in the background and sends a callback. Used for "Async" mode.
 * The core logic is parallelized for maximum speed.
 */
async function processSummaryAsync({
    accountId, accessToken, callbackUrl, userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId, finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId, sendCallback
}) {
    const conn = new jsforce.Connection({ instanceUrl: SF_LOGIN_URL, accessToken: accessToken, maxRequest: 5, version: '59.0' });

    try {
        // 1. Fetch data
        console.log(`[${accountId}] (Async) Fetching Salesforce records...`);
        const groupedData = await fetchRecords(conn, queryText);

        // 2. Generate ALL monthly summaries in PARALLEL
        const { finalMonthlySummaries, monthlyGenerationErrors } = await generateAllMonthlySummaries(
            groupedData, accountId, userPromptMonthlyTemplate, finalMonthlyAssistantId, finalMonthlyFuncSchema
        );

        if (monthlyGenerationErrors.length > 0) {
             console.warn(`[${accountId}] (Async) Encountered ${monthlyGenerationErrors.length} errors during monthly summary generation. Process will continue.`);
        }

        // 3. Save monthly summaries to SF
        await saveSummariesToSalesforce(conn, finalMonthlySummaries, accountId, 'Monthly', summaryRecordsMap, loggedinUserId);

        // 4. Generate ALL quarterly summaries in PARALLEL
        const { finalQuarterlyDataForSalesforce, quarterlyGenerationErrors } = await generateAllQuarterlySummaries(
            finalMonthlySummaries, accountId, userPromptQuarterlyTemplate, finalQuarterlyAssistantId, finalQuarterlyFuncSchema
        );

        if (quarterlyGenerationErrors.length > 0) {
             console.warn(`[${accountId}] (Async) Encountered ${quarterlyGenerationErrors.length} errors during quarterly summary generation. Process will continue.`);
        }
        
        // 5. Save quarterly summaries to SF
        await saveSummariesToSalesforce(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId);

        // 6. Send Success Callback
        console.log(`[${accountId}] (Async) Process completed successfully.`);
        if (sendCallback === 'Yes') {
            const message = `Summary Processed Successfully. Monthly Errors: ${monthlyGenerationErrors.length}. Quarterly Errors: ${quarterlyGenerationErrors.length}.`;
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", message);
        }

    } catch (error) {
        console.error(`[${accountId}] (Async) Error during summary processing:`, error);
        if (sendCallback === 'Yes') {
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Processing error: ${error.message}`);
        }
    }
}


// --- PARALLELIZED SUMMARY GENERATION LOGIC ---

async function generateAllMonthlySummaries(groupedData, accountId, userPromptMonthlyTemplate, assistantId, schema) {
    const finalMonthlySummaries = {};
    const monthlyPromises = [];
    const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };

    for (const year in groupedData) {
        finalMonthlySummaries[year] = {};
        for (const monthObj of groupedData[year]) {
            for (const month in monthObj) {
                const activities = monthObj[month];
                if (activities.length === 0) continue;

                const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);
                // Push the promise AND the metadata needed to place the result later
                monthlyPromises.push({
                    promise: generateSummary(activities, openai, assistantId, userPromptMonthly, schema),
                    year,
                    month
                });
            }
        }
    }

    if (monthlyPromises.length === 0) {
        console.log(`[${accountId}] No months with activities found to process.`);
        return { finalMonthlySummaries: {}, monthlyGenerationErrors: [] };
    }

    console.log(`[${accountId}] Generating ${monthlyPromises.length} monthly summaries in PARALLEL...`);
    const results = await Promise.allSettled(monthlyPromises.map(p => p.promise));

    const monthlyGenerationErrors = [];
    results.forEach((result, index) => {
        const { year, month } = monthlyPromises[index];
        if (result.status === 'fulfilled') {
            const monthIndex = monthMap[month.toLowerCase()];
            const startDate = new Date(Date.UTC(year, monthIndex, 1));
            finalMonthlySummaries[year][month] = {
                aiOutput: result.value,
                count: groupedData[year].find(m => m[month])[month].length,
                startdate: startDate.toISOString().split('T')[0],
                year: parseInt(year),
                monthIndex: monthIndex
            };
            console.log(`[${accountId}]   SUCCESS: Generated summary for ${month} ${year}.`);
        } else {
            console.error(`[${accountId}]   FAILED: Generating summary for ${month} ${year}. Reason:`, result.reason);
            monthlyGenerationErrors.push({ period: `${month} ${year}`, error: result.reason.message || result.reason });
        }
    });

    return { finalMonthlySummaries, monthlyGenerationErrors };
}

async function generateAllQuarterlySummaries(finalMonthlySummaries, accountId, userPromptQuarterlyTemplate, assistantId, schema) {
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

    const quarterlyPromises = [];
    for (const [quarterKey, monthlySummariesForQuarter] of Object.entries(quarterlyInputGroups)) {
        if (!monthlySummariesForQuarter || monthlySummariesForQuarter.length === 0) continue;

        const [year, quarter] = quarterKey.split('-');
        const quarterlyInputDataString = JSON.stringify(monthlySummariesForQuarter, null, 2);
        const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly summary data for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

        quarterlyPromises.push({
            promise: generateSummary(null, openai, assistantId, userPromptQuarterly, schema),
            quarterKey
        });
    }

    if (quarterlyPromises.length === 0) {
        console.log(`[${accountId}] No quarters with data found to process.`);
        return { finalQuarterlyDataForSalesforce: {}, quarterlyGenerationErrors: [] };
    }

    console.log(`[${accountId}] Generating ${quarterlyPromises.length} quarterly summaries in PARALLEL...`);
    const results = await Promise.allSettled(quarterlyPromises.map(p => p.promise));

    const finalQuarterlyDataForSalesforce = {};
    const quarterlyGenerationErrors = [];
    results.forEach((result, index) => {
        const { quarterKey } = quarterlyPromises[index];
        if (result.status === 'fulfilled') {
            const transformedResult = transformQuarterlyStructure(result.value);
            for (const year in transformedResult) {
                if (!finalQuarterlyDataForSalesforce[year]) finalQuarterlyDataForSalesforce[year] = {};
                Object.assign(finalQuarterlyDataForSalesforce[year], transformedResult[year]);
            }
            console.log(`[${accountId}]   SUCCESS: Generated summary for ${quarterKey}.`);
        } else {
            console.error(`[${accountId}]   FAILED: Generating summary for ${quarterKey}. Reason:`, result.reason);
            quarterlyGenerationErrors.push({ period: quarterKey, error: result.reason.message || result.reason });
        }
    });

    return { finalQuarterlyDataForSalesforce, quarterlyGenerationErrors };
}

async function saveSummariesToSalesforce(conn, summaries, parentId, summaryCategory, summaryRecordsMap, loggedinUserId) {
    // This is a wrapper for the already-bulk-capable createTimileSummarySalesforceRecords function
    const sfPayload = {};
    for (const year in summaries) {
        sfPayload[year] = {};
        for (const period in summaries[year]) {
            const data = summaries[year][period];
            // Normalize structure for the save function
            sfPayload[year][period] = {
                summary: JSON.stringify(data.aiOutput || data.summaryJson),
                summaryDetails: data.aiOutput?.summary || data.summaryDetails || '',
                count: data.count,
                startdate: data.startdate
            };
        }
    }

    if (Object.keys(sfPayload).length > 0 && Object.values(sfPayload).some(year => Object.keys(year).length > 0)) {
        console.log(`[${parentId}] Saving ${summaryCategory} summaries to Salesforce...`);
        await createTimileSummarySalesforceRecords(conn, sfPayload, parentId, summaryCategory, summaryRecordsMap, loggedinUserId);
        console.log(`[${parentId}] ${summaryCategory} summaries saved.`);
    } else {
        console.log(`[${parentId}] No ${summaryCategory} summaries generated to save.`);
    }
}


// --- CORE HELPER FUNCTIONS (Largely unchanged, but with minor logging improvements) ---

// OpenAI Summary Generation Function (Unchanged)
async function generateSummary(activities, openaiClient, assistantId, userPrompt, functionSchema) {
    let fileId = null, thread = null, filePath = null;
    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        thread = await openaiClient.beta.threads.create();
        let finalUserPrompt = userPrompt, messageAttachments = [], inputMethod = "prompt";

        if (activities && activities.length > 0) {
            const activitiesJsonString = JSON.stringify(activities, null, 2);
            const potentialFullPrompt = `${userPrompt}\n\nHere is the activity data:\n\`\`\`json\n${activitiesJsonString}\n\`\`\``;

            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && activities.length <= DIRECT_INPUT_THRESHOLD) {
                finalUserPrompt = potentialFullPrompt;
                inputMethod = "direct JSON";
            } else {
                inputMethod = "file upload";
                const activitiesText = activities.map((act, i) => `Activity ${i+1}:\n${Object.entries(act).map(([k,v]) => `  ${k}: ${JSON.stringify(v)}`).join('\n')}`).join('\n\n---\n\n');
                filePath = path.join(TEMP_FILE_DIR, `activities_${thread.id}.txt`);
                await fs.writeFile(filePath, activitiesText);
                const uploadResponse = await openaiClient.files.create({ file: fs.createReadStream(filePath), purpose: "assistants" });
                fileId = uploadResponse.id;
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
            }
        }
        
        console.log(`[Thread ${thread.id}] Creating message using ${inputMethod} for Asst ${assistantId}`);
        await openaiClient.beta.threads.messages.create(thread.id, { role: "user", content: finalUserPrompt, attachments: messageAttachments });
        
        console.log(`[Thread ${thread.id}] Starting run, forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }],
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });

        if (run.status === 'requires_action') {
            const toolCall = run.required_action?.submit_tool_outputs?.tool_calls[0];
            if (!toolCall) throw new Error("Function call was expected but not provided by the Assistant.");
            console.log(`[Thread ${thread.id}] Function call received. Parsing arguments.`);
            return JSON.parse(toolCall.function.arguments);
        } else {
            console.error(`[Thread ${thread.id}] Run ended unexpectedly. Status: ${run.status}`, run.last_error);
            const errorMsg = run.last_error ? `${run.last_error.code}: ${run.last_error.message}` : `Run status was ${run.status}`;
            throw new Error(`Assistant run failed. ${errorMsg}`);
        }
    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary: ${error.message}`, error);
        throw error;
    } finally {
        if (filePath) await fs.unlink(filePath).catch(e => console.error(`Error deleting temp file ${filePath}:`, e));
        if (fileId) await openaiClient.files.del(fileId).catch(e => { if (!(e instanceof NotFoundError)) console.error(`Error deleting OpenAI file ${fileId}:`, e); });
    }
}

// Salesforce Record Creation/Update Function (Unchanged, already uses bulk-capable methods)
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap, loggedinUserId) {
    let recordsToCreate = [], recordsToUpdate = [];
    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const summaryData = summaries[year][periodKey];
            let summaryJsonString = summaryData.summary;
            let summaryDetailsHtml = summaryData.summaryDetails;
            if (!summaryDetailsHtml && summaryJsonString) {
                try { summaryDetailsHtml = JSON.parse(summaryJsonString)?.summary || ''; } catch (e) { /* ignore */ }
            }
            const recordPayload = {
                Parent_Id__c: parentId,
                Month__c: summaryCategory === 'Monthly' ? periodKey : null,
                Year__c: String(year),
                Summary_Category__c: summaryCategory,
                Requested_By__c: loggedinUserId,
                Type__c: 'Activity',
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null,
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null,
                FY_Quarter__c: summaryCategory === 'Quarterly' ? periodKey : null,
                Month_Date__c: summaryData.startdate,
                Number_of_Records__c: summaryData.count || 0,
            };
            const summaryMapKey = summaryCategory === 'Quarterly' ? `${periodKey} ${year}` : `${periodKey.substring(0, 3)} ${year}`;
            const existingRecordId = getValueByKey(summaryRecordsMap, summaryMapKey);
            if (existingRecordId) {
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                recordsToCreate.push(recordPayload);
            }
        }
    }
    try {
        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} summary records in a batch...`);
            const createResults = await conn.sobject(TIMELINE_SUMMARY_OBJECT_API_NAME).create(recordsToCreate, { allOrNone: false });
            handleBulkResults(createResults, 'create', parentId);
        }
        if (recordsToUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${recordsToUpdate.length} existing ${summaryCategory} summary records in a batch...`);
            const updateResults = await conn.sobject(TIMELINE_SUMMARY_OBJECT_API_NAME).update(recordsToUpdate, { allOrNone: false });
            handleBulkResults(updateResults, 'update', parentId);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} records to Salesforce: ${err.message}`, err);
        throw new Error(`Salesforce save operation failed: ${err.message}`);
    }
}

// Salesforce Data Fetching with Pagination (Unchanged)
async function fetchRecords(conn, queryOrUrl, allRecords = [], isFirstIteration = true) {
    try {
        const queryResult = isFirstIteration ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);
        if (queryResult.records && queryResult.records.length > 0) {
            allRecords.push(...queryResult.records);
        }
        console.log(`[SF Fetch] Fetched ${queryResult.records.length}. Total so far: ${allRecords.length}. Done: ${queryResult.done}`);
        if (!queryResult.done && queryResult.nextRecordsUrl) {
            await new Promise(resolve => setTimeout(resolve, 200));
            return fetchRecords(conn, queryResult.nextRecordsUrl, allRecords, false);
        }
        console.log(`[SF Fetch] Finished. Total records: ${allRecords.length}. Grouping...`);
        return groupRecordsByMonthYear(allRecords);
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce activities: ${error.message}`, error);
        throw error;
    }
}

// Data Grouping Helper (Unchanged)
function groupRecordsByMonthYear(records) {
    const groupedData = {};
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    records.forEach(activity => {
        if (!activity.CreatedDate) return;
        const date = new Date(activity.CreatedDate);
        if (isNaN(date.getTime())) return;
        const year = date.getUTCFullYear();
        const month = monthNames[date.getUTCMonth()];
        if (!groupedData[year]) groupedData[year] = [];
        let monthEntry = groupedData[year].find(entry => entry[month]);
        if (!monthEntry) {
            monthEntry = { [month]: [] };
            groupedData[year].push(monthEntry);
        }
        monthEntry[month].push({ Id: activity.Id, Description: activity.Description, Subject: activity.Subject, CreatedDate: activity.CreatedDate });
    });
    return groupedData;
}

// Callback Sending Function (Unchanged)
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}`);
    try {
        await axios.post(callbackUrl, {
            accountId: accountId, loggedinUserId: loggedinUserId, status: "Completed",
            processResult: status, message: message
        }, {
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${accessToken}` },
            timeout: 30000
        });
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        console.error(`[${accountId}] Failed to send callback to ${callbackUrl}: ${error.message}`);
    }
}

// --- UTILITY HELPER FUNCTIONS (Unchanged and still used) ---

function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex <= 2) return 'Q1';
    if (monthIndex <= 5) return 'Q2';
    if (monthIndex <= 8) return 'Q3';
    return 'Q4';
}

function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray) return null;
    const record = recordsArray.find(item => item && item.key.toLowerCase() === searchKey.toLowerCase());
    return record ? record.value : null;
}

function handleBulkResults(results, operationType, parentId) {
    const successes = results.filter(r => r.success).length;
    const failures = results.length - successes;
    console.log(`[${parentId}] Bulk ${operationType} summary: ${successes} succeeded, ${failures} failed.`);
    if (failures > 0) {
        results.filter(r => !r.success).forEach(res => {
            console.error(`[${parentId}]   Error on record ID ${res.id}: ${JSON.stringify(res.errors)}`);
        });
    }
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
    if (!year || !quarter) return result;
    const getQuarterStartMonth = (q) => ({ 'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10' })[q.toUpperCase()] || '01';
    result[year] = {
        [quarter]: {
            summaryDetails: quarterData.summary || '',
            summaryJson: JSON.stringify(quarterData),
            count: quarterData.activityCount || 0,
            startdate: quarterData.startdate || `${year}-${getQuarterStartMonth(quarter)}-01`
        }
    };
    return result;
}
