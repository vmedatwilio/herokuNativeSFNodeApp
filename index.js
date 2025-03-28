const express = require('express');
const jsforce = require('jsforce');
const dotenv = require('dotenv');
const { OpenAI } = require("openai");
const fs = require("fs-extra");
const path = require("path");
const axios = require("axios");
dotenv.config();
const app = express();
const SF_LOGIN_URL = process.env.SF_LOGIN_URL;

const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});


// STEP 1: Async Function to Process Summary
app.post('/generatesummary', async (req, res) => {

    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ error: "Unauthorized" });
    }

    const accessToken = authHeader.split(" ")[1];
    //console.log(accessToken);
    const { accountId, callbackUrl, userPrompt,userPromptQtr, queryText, summaryMap, loggedinUserId } = req.body;
        
        if (!accountId  || !callbackUrl || !accessToken) {
            return res.status(400).send({ error: "Missing required parameters" });
        }
        res.json({ status: 'processing', message: 'Summary is being generated' });
        let summaryRecordsMap={};
        if(summaryMap != undefined && summaryMap != null && summaryMap != '') {
            summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        }

        processSummary(accountId, accessToken, callbackUrl, userPrompt,  userPromptQtr, queryText, (summaryMap != undefined && summaryMap != null && summaryMap != '') ? summaryRecordsMap : null, loggedinUserId);
});

async function sendCallbackResponse(accountId,callbackUrl,loggedinUserId, accessToken, status, message) {
    console.log('callback message ::: '+ message);
    await axios.post(callbackUrl, 
        {
            accountId : accountId,
            loggedinUserId : loggedinUserId,
            status: "Completed",
            processResult: status,
            message
        }, 
        {
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${accessToken}`
            }
        }
    );
}

// Helper function to process summary generation asynchronously
async function processSummary(accountId, accessToken, callbackUrl, userPrompt,userPromptQtr, queryText, summaryRecordsMap, loggedinUserId) {

    try {
        
        const conn = new jsforce.Connection({
            instanceUrl: SF_LOGIN_URL,
            accessToken: accessToken
        });
       
        //let queryStr = `SELECT Description,ActivityDate FROM Task WHERE ActivityDate!=null and AccountId = '${accountId}' AND ActivityDate >= LAST_N_YEARS:4 ORDER BY ActivityDate DESC`;
        
        const groupedData = await fetchRecords(conn, queryText);

        //Step 1: intiate Open AI
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY, // Read from .env
          });
        
        // Step 2: Create an Assistant (if not created before)
        const assistant = await openai.beta.assistants.create({
            name: "Salesforce Summarizer",
            instructions: "You are an AI that summarizes Salesforce activity data.",
            tools: [{ type: "file_search" }], // Allows using files
            model: "gpt-4-turbo",
        });

        const finalSummary = {};

        const monthMap = {
            january: 0,
            february: 1,
            march: 2,
            april: 3,
            may: 4,
            june: 5,
            july: 6,
            august: 7,
            september: 8,
            october: 9,
            november: 10,
            december: 11
        };

        // let userPrompt = `You are an AI assistant generating structured sales activity summaries for Salesforce.
        //                     ### **Instructions**
        //                     - Analyze the provided sales activity data and generate a **monthly summary {{YearMonth}}**.
        //                     - Extract the **key themes of customer interactions** based on email content.
        //                     - Describe the **tone and purpose** of the interactions.
        //                     - Identify any **response trends** and suggest relevant **follow-up actions**.
        //                     - Format the summary in **HTML** suitable for a Salesforce **Rich Text Area field**.
        //                     - Return **only the formatted summary** without explanations.
        //                     ### **Formatting Instructions**
        //                     - **Use only these HTML tags** (<b>, <br>, <ul><li>) for structured formatting.
        //                     - **Do not include explanations**—return only the final HTML summary.
        //                     - **Ensure readability and clarity.**`;

        for (const year in groupedData) {
            console.log(`Year: ${year}`);
            finalSummary[year] = {};
            // Iterate through the months inside each year
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    console.log(`  Month: ${month}`);
                    const tmpactivites = monthObj[month];
                    console.log(`  ${month}: ${tmpactivites.length} activities`);
                    const monthIndex = monthMap[month.toLowerCase()];
                    const startdate = new Date(year, monthIndex, 1);
                    const summary = await generateSummary(tmpactivites,openai,assistant,userPrompt.replace('{{YearMonth}}',`${month} ${year}`));
                    finalSummary[year][month] = {"summary":summary,"count":tmpactivites.length,"startdate":startdate};
                }
            }
        }

        const createmonthlysummariesinsalesforce = await createTimileSummarySalesforceRecords(conn, finalSummary,accountId,'Monthly',summaryRecordsMap);

        const quarterlyPrompt = `Using the provided monthly summary data, generate a consolidated quarterly summary for each year. Each quarter should combine insights from its respective months (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec).
                                    Return a valid, parseable JSON object with this exact structure:
                                    {
                                    "YEAR": {
                                        "Q1": {
                                        "summary": "Consolidated quarterly summary text",
                                        "count": NUMERIC_SUM_OF_MONTHLY_COUNTS,
                                        "startdate": "YYYY-MM-DD"
                                        },
                                        "Q2": { ... },
                                        "Q3": { ... },
                                        "Q4": { ... }
                                    },
                                    "YEAR2": { ... }
                                    }
                                    **Strict requirements:**
                                    1. Ensure all property names use double quotes
                                    2. Format dates as ISO strings (YYYY-MM-DD)
                                    3. The "count" field must be a number, not a string, and only add the summary if count > 0 for a quarter, if it is 0 remove this quarter from json
                                    4. The "startdate" should be the first day of the quarter (Jan 1, Apr 1, Jul 1, Oct 1)
                                    5. Return only the raw JSON with no explanations or formatting
                                    6. Ensure the JSON is minified (no extra spaces or line breaks)
                                    7. Each quarter should have exactly these three properties: summary, count, startdate
                                    8. **Ensure JSON is in minified format** (i.e., no extra spaces, line breaks, or special characters).
                                    9. The response **must be directly usable with "JSON.parse(response)"**.
                                    10.**Return only the raw JSON object** with no explanations, Markdown formatting, or extra characters. Do not wrap the JSON in triple backticks or include "json" as a specifier.`;


        const Quarterlysummary = await generateSummary(finalSummary,openai,assistant,userPromptQtr);
                          

        const quaertersums=JSON.parse(Quarterlysummary);
        console.log(`Quarterlysummary received ${JSON.stringify(quaertersums)}`);

        const createQuarterlysummariesinsalesforce = await createTimileSummarySalesforceRecords(conn,quaertersums,accountId,'Quarterly',summaryRecordsMap);
        await sendCallbackResponse(accountId,callbackUrl,loggedinUserId, accessToken, "Success", "Summary Processed Successfully"); 

    } catch (error) {
        console.error(error);
        await sendCallbackResponse(accountId,callbackUrl,loggedinUserId, accessToken, "Failed", error.message);
    }

}

async function createTimileSummarySalesforceRecords( conn,summaries={},parentId,summaryCategory,summaryRecordsMap) {

    // Create a unit of work that inserts multiple objects.
    let recordsToCreate =[];
    let recordsToUpdate =[];
        
    for (const year in summaries) {
        //logger.info(`Year: ${year}`);
        for (const month in summaries[year]) {
            //logger.info(`Month: ${month}`);
            //logger.info(`Summary:\n${summaries[year][month].summary}\n`);
            let FYQuartervalue=(summaryCategory=='Quarterly')?month:'';
            let motnhValue=(summaryCategory=='Monthly')?month:'';
            let shortMonth = motnhValue.substring(0, 3);
            let summaryValue=summaries[year][month].summary;
            let startdate=summaries[year][month].startdate;
            let count=summaries[year][month].count;

            let summaryMapKey = (summaryCategory=='Quarterly')? FYQuartervalue + ' ' + year : shortMonth + ' ' + year;
            let recId = (summaryRecordsMap!=null) ? getValueByKey(summaryRecordsMap,summaryMapKey):null;

             // Push record to the list
             if(recId!=null && recId!=undefined) {
                recordsToUpdate.push({
                    Id: recId,
                    Parent_Id__c: parentId,
                    Month__c: motnhValue,
                    Year__c: year,
                    Summary_Category__c: summaryCategory,
                    Summary_Details__c: summaryValue,
                    FY_Quarter__c: FYQuartervalue,
                    Month_Date__c: startdate,
                    Number_of_Records__c: count,
                    Account__c: parentId
                });
             }
             else {
                recordsToCreate.push({
                    Parent_Id__c: parentId,
                    Month__c: motnhValue,
                    Year__c: year,
                    Summary_Category__c: summaryCategory,
                    Summary_Details__c: summaryValue,
                    FY_Quarter__c: FYQuartervalue,
                    Month_Date__c: startdate,
                    Number_of_Records__c: count,
                    Account__c: parentId
                });
             }
        }
    }
    try {
        // Commit the Unit of Work with all the previous registered operations
        if (recordsToCreate.length > 0) {
            // Insert all records at once
            const result = await conn.sobject("Timeline_Summary__c").insert(recordsToCreate);

            // Handle response
            result.forEach((res, index) => {
                if (res.success) {
                    console.log(`Record ${index + 1} inserted with Id: ${res.id}`);
                } else {
                    console.error(`Error inserting record ${index + 1}:`, res.errors);
                }
            });
        } else {
            console.log("No records to insert.");
        }
        if (recordsToUpdate.length > 0) {
            // Insert all records at once
            const result = await conn.sobject("Timeline_Summary__c").update(recordsToUpdate);

            // Handle response
            result.forEach((res, index) => {
                if (res.success) {
                    console.log(`Record ${index + 1} Updated with Id: ${res.id}`);
                } else {
                    console.error(`Error updating record ${index + 1}:`, res.errors);
                }
            });
        } else {
            console.log("No records to update.");
        }
    }
    catch (err) {
        const errorMessage = `Failed to update record. Root Cause : ${err.message}`;
        console.error(errorMessage);
        throw new Error(errorMessage);
    }
}

//funtcion to generate summary from OpenAI
async function generateSummary(activities, openai,assistant,userPrompt) 
{
    try 
    {
        // Step 1: Generate JSON file
        const filePath = await generateFile(activities);

        // Step 2: Upload file to OpenAI
        const uploadResponse = await openai.files.create({
            file: fs.createReadStream(filePath),
            purpose: "assistants", // Required for storage
        });
            
        const fileId = uploadResponse.id;
        console.log(`File uploaded to OpenAI: ${fileId}`);

        // Step 4: Create a Thread
        const thread = await openai.beta.threads.create();
        console.log(`Thread created: ${thread.id}`);

        // Step 5: Submit Message to Assistant (referencing file)
        const message = await openai.beta.threads.messages.create(thread.id, {
            role: "user",
            content:userPrompt,
                    attachments: [
                        { 
                            file_id: fileId,
                            tools: [{ type: "file_search" }],
                        }
                    ],
                });
            
        console.log(`Message sent: ${message.id}`);

        // Step 6: Run the Assistant
        const run = await openai.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistant.id,
        });
            
        console.log(`Run started: ${run.id}`);

        const messages = await openai.beta.threads.messages.list(thread.id, {
            run_id: run.id,
          });

          // Log the full response structure
          console.log(`OpenAI msg content Response: ${JSON.stringify(messages, null, 2)}`);

          const summary = messages.data[0].content[0].text.value;
          console.log(`Summary received ${JSON.stringify(messages.data[0].content[0])}`);
        
          console.log(`Summary received ${summary}`);

          const file = await openai.files.del(fileId);

          console.log(file);

        return summary.replace(/(\[\[\d+†source\]\]|\【\d+:\d+†source\】)/g, '');
    } 
    catch (error) 
    {
        console.error(`Error generating summary : ${error.message}`);
        throw error;
    }
}

async function generateFile( activities = []) {

    // Get current date-time in YYYYMMDD_HHMMSS format
    const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
    const filename = `salesforce_activities_${timestamp}.json`;

    const filePath = path.join(__dirname, filename);
    try {
        //const jsonlData = activities.map((entry) => JSON.stringify(entry)).join("\n");
        await fs.writeFile(filePath, JSON.stringify(activities, null, 2), "utf-8");
        //await fs.writeFile(filePath, jsonlData, "utf-8");
        console.log(`File Generated successfully ${filePath}`);
        return filePath;
    } catch (error) {
        console.log(`Error writing file: ${error}`);
        throw error;
    }
}

// Function to fetch records recursively and group them
async function fetchRecords(conn, queryOrUrl, groupedData = {}, isFirstIteration = true) {
    try {
        // Query Salesforce (initial query or queryMore for pagination)
        console.log('query is ::: '+queryOrUrl);
        const queryResult = isFirstIteration ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);

        queryResult.records.forEach(activity => {
            const date = new Date(activity.ActivityDate); // Ensure this field exists in the query
            const year = date.getFullYear();
            const month = date.toLocaleString('en-US', { month: 'long' });

            if (!groupedData[year]) {
                groupedData[year] = [];
            }

            // Find or create the month entry
            let monthEntry = groupedData[year].find(entry => entry[month]);
            if (!monthEntry) {
                monthEntry = { [month]: [] };
                groupedData[year].push(monthEntry);
            }

            //monthEntry[month].push(activity.Description || "No Description"); // Change field if needed

            monthEntry[month].push({
                Id: activity.Id,
                Description: activity.Description || "No Description",
                Subject: activity.Subject || "No Subject"
            });
        });

        // If there are more records, fetch them recursively
        if (queryResult.nextRecordsUrl) {
            return fetchRecords(conn, queryResult.nextRecordsUrl, groupedData, false);
        } else {
            return groupedData;
        }
    } catch (error) {
        console.error(`Error fetching activities: ${error.message}`);
        throw error;
    }
    
}
function getValueByKey(records, searchKey) {
    const record = records.find(item => item.key === searchKey);
    return record ? record.value : null;
}