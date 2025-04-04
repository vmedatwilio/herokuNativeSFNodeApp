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

const functions = [

    {
        "name": "generate_monthly_activity_summary",
        "description": "Sales activity summary generator with structured insights and categorization",
        "parameters": {
          "type": "object",
          "properties": {
            "summary": {
              "type": "string",
              "description": "Descriptive summary of all activities grouped monthly based on activity date, make sure to include activties only with the current month activitydate, highlighting key trends and patterns, should be strictly in HTML rich text format having one header (strictly only within <h1> tag no bold) 'Sales Activity Summary for {Month} {Year}' and multiple bullet points containing key insights"
            },
            "activityMapping": {
              "type": "object",
              "description": "Detailed activity summary and categorization with strategic mapping",
              "properties": {
                "Key Themes of Customer Interaction": {
                  "type": "array",
                  "description": "Strategic insights into the major themes of customer communication patterns with deep, actionable subcategories, analyze the provided prompt for detailed requirement",
                  "items": {
                    "type": "object",
                    "properties": {
                      "Summary": { "type": "string", "description": "Descriptive summary of the activities coming under this criteria" },
                      "ActivityList": {
                        "type": "array",
                        "description": "List of activities belonging to this header that has activity date in this month",
                        "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific task belonging to this header" },
                            "LinkText": { "type": "string", "description": "Combination of 'Activity Date' : 'Short Description of the activity description'" }
                          },
                          "required": ["Id", "LinkText"],
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
                  "description": "Nuanced communication dynamics and strategic intent analysis, analyze the provided prompt for detailed requirement",
                  "items": {
                    "type": "object",
                    "properties": {
                      "Summary": { "type": "string", "description": "Descriptive summary of the activities coming under this criteria" },
                      "ActivityList": {
                        "type": "array",
                        "description": "List of activities belonging to this header that has activity date in this month",
                        "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific task belonging to this header" },
                            "LinkText": { "type": "string", "description": "Combination of 'Activity Date' : 'Short Description of the activity description'" }
                          },
                          "required": ["Id", "LinkText"],
                          "additionalProperties": false
                        }
                      }
                    },
                    "required": ["Summary", "ActivityList"],
                    "additionalProperties": false
                  }
                },
                "Recommended Action and Next Steps": {
                  "type": "array",
                  "description": "Forward-looking strategic recommendations with executable guidance, analyze the provided prompt for detailed requirement",
                  "items": {
                    "type": "object",
                    "properties": {
                      "Summary": { "type": "string", "description": "Descriptive summary of the activities coming under this criteria" },
                      "ActivityList": {
                        "type": "array",
                        "description": "List of activities belonging to this header that has activity date in this month",
                        "items": {
                          "type": "object",
                          "properties": {
                            "Id": { "type": "string", "description": "Salesforce Id of the specific task belonging to this header" },
                            "LinkText": { "type": "string", "description": "Combination of 'Activity Date' : 'Short Description of the activity description'" }
                          },
                          "required": ["Id", "LinkText"],
                          "additionalProperties": false
                        }
                      }
                    },
                    "required": ["Summary", "ActivityList"],
                    "additionalProperties": false
                  }
                }
              },
              "required": ["Key Themes of Customer Interaction", "Tone and Purpose of Interaction", "Recommended Action and Next Steps"],
              "additionalProperties": false
            },
            "activityCount": {
              "type": "integer",
              "description": "Total number of incoming sales activities for the month based on activity Date in JSON input file"
            }
          },
          "required": ["summary", "activityMapping", "activityCount"],
          "additionalProperties": false
        },
        "strict": true
      },
      {
        "name": "generate_quarterly_activity_summary",
        "description": "Sales activity summary generator structured dynamically by year and quarter, including key insights and activity categorization",
        "parameters": {
          "type": "object",
          "properties": {
            "yearlySummary": {
              "type": "array",
              "description": "Structured summary of activities grouped dynamically by year and quarter",
              "items": {
                "type": "object",
                "properties": {
                  "year": {
                    "type": "integer",
                    "description": "Year of the sales activity summary"
                  },
                  "quarters": {
                    "type": "array",
                    "description": "List of quarterly summaries for the given year",
                    "items": {
                      "type": "object",
                      "properties": {
                        "quarter": {
                          "type": "string",
                          "description": "Quarter identifier (e.g., Q1, Q2, Q3, Q4), Strictly all quarters of a year where activities are present should be present"
                        },
                        "summary": {
                          "type": "string",
                          "description": "Descriptive summary of activities for the quarter, in HTML format, with header  (strictly only within <h1> tag no bold) 'Sales Activity Summary for {Quarter} {Year}' and key insights as bullet points"
                        },
                        "activityMapping": {
                          "type": "array",
                          "description": "List of categorized activities with descriptions and related tasks",
                          "items": {
                            "type": "object",
                            "properties": {
                              "category": {
                                "type": "string",
                                "description": "Category of activities (e.g., Key Themes of Customer Interaction, Tone and Purpose of Interaction, Recommended Action and Next Steps)"
                              },
                              "summary": {
                                "type": "string",
                                "description": "Descriptive summary of activities in this category"
                              },
                              "activityList": {
                                "type": "array",
                                "items": {
                                  "type": "object",
                                  "properties": {
                                    "id": { "type": "string", "description": "Salesforce ID of the task" },
                                    "linkText": { "type": "string", "description": "'Activity Date' : 'Short Description'" }
                                  },
                                  "required": ["id", "linkText"],
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
                          "description": "Total number of sales activities for the quarter"
                        },
                        "count": {
                          "type": "integer",
                          "description": "Numeric sum of activities recorded in the quarter"
                        },
                        "startdate": {
                          "type": "string",
                          "description": "Start date of the quarter in YYYY-MM-DD format"
                        }
                      },
                      "required": ["quarter", "summary", "activityMapping", "activityCount", "count", "startdate"],
                      "additionalProperties": false
                    }
                  }
                },
                "required": ["year", "quarters"],
                "additionalProperties": false
              }
            }
          },
          "required": ["yearlySummary"],
          "additionalProperties": false
        },
        "strict": true
      }
      
      
      
      
      
  ];
  
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
    const { accountId, callbackUrl, userPrompt,userPromptQtr, queryText, summaryMap, loggedinUserId,qtrJSON,monthJSON } = req.body;
        
        if (!accountId  || !callbackUrl || !accessToken) {
            return res.status(400).send({ error: "Missing required parameters" });
        }
        res.json({ status: 'processing', message: 'Summary is being generated' });
        let summaryRecordsMap={};
        if(summaryMap != undefined && summaryMap != null && summaryMap != '') {
            summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        }

        processSummary(accountId, accessToken, callbackUrl, userPrompt,  userPromptQtr, queryText, (summaryMap != undefined && summaryMap != null && summaryMap != '') ? summaryRecordsMap : null, loggedinUserId,qtrJSON,monthJSON);
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
async function processSummary(accountId, accessToken, callbackUrl, userPrompt,userPromptQtr, queryText, summaryRecordsMap, loggedinUserId,qtrJSON,monthJSON) {

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
            name: "Salesforce Monthly Summarizer",
            instructions: "You are an AI assistant specialized in analyzing raw Salesforce activity data for a single month and generating structured JSON summaries using the provided function 'generate_monthly_activity_summary'. Focus on extracting key themes, tone, and recommended actions.",
            tools: [{ type: "file_search" },{type:"function" , "function" : monthJSON!= undefined ? JSON.parse(monthJSON) : functions[0]}], // Allows using files
            model: "gpt-4o",
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
                    finalSummary[year][month] = {"summary":JSON.stringify(summary),"count":tmpactivites.length,"startdate":startdate};
                }
            }
        }

        const createmonthlysummariesinsalesforce = await createTimileSummarySalesforceRecords(conn, finalSummary,accountId,'Monthly',summaryRecordsMap);

        // const quarterlyPrompt = `Using the provided monthly summary data, generate a consolidated quarterly summary for each year. Each quarter should combine insights from its respective months (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec).
        //                             Return a valid, parseable JSON object with this exact structure:
        //                             {
        //                             "YEAR": {
        //                                 "Q1": {
        //                                 "summary": "Consolidated quarterly summary text",
        //                                 "count": NUMERIC_SUM_OF_MONTHLY_COUNTS,
        //                                 "startdate": "YYYY-MM-DD"
        //                                 },
        //                                 "Q2": { ... },
        //                                 "Q3": { ... },
        //                                 "Q4": { ... }
        //                             },
        //                             "YEAR2": { ... }
        //                             }
        //                             **Strict requirements:**
        //                             1. Ensure all property names use double quotes
        //                             2. Format dates as ISO strings (YYYY-MM-DD)
        //                             3. The "count" field must be a number, not a string, and only add the summary if count > 0 for a quarter, if it is 0 remove this quarter from json
        //                             4. The "startdate" should be the first day of the quarter (Jan 1, Apr 1, Jul 1, Oct 1)
        //                             5. Return only the raw JSON with no explanations or formatting
        //                             6. Ensure the JSON is minified (no extra spaces or line breaks)
        //                             7. Each quarter should have exactly these three properties: summary, count, startdate
        //                             8. **Ensure JSON is in minified format** (i.e., no extra spaces, line breaks, or special characters).
        //                             9. The response **must be directly usable with "JSON.parse(response)"**.
        //                             10.**Return only the raw JSON object** with no explanations, Markdown formatting, or extra characters. Do not wrap the JSON in triple backticks or include "json" as a specifier.`;


        const assistantQtr = await openai.beta.assistants.create({
          name: "Salesforce Quarterly Aggregator",
          instructions: "You are an AI assistant specialized in aggregating pre-summarized monthly Salesforce activity data (provided as JSON in the prompt) into a structured quarterly JSON summary using the provided function 'generate_quarterly_activity_summary'. Consolidate insights and activity lists accurately.",
          tools: [{ type: "file_search" },{type:"function" , "function" : qtrJSON!= undefined ? JSON.parse(qtrJSON) : functions[1]}], // Allows using files
          model: "gpt-4o",
      });

        const Quarterlysummary = await generateSummary(finalSummary,openai,assistantQtr,userPromptQtr);
                          

        //const quaertersums= Quarterlysummary;
        console.log(`Quarterlysummary received ${JSON.stringify(Quarterlysummary)}`);
        // Transform the structure to match the required format
        const quaertersums = transformStructure(Quarterlysummary);
        console.log(`Transformed Quarterlysummary: ${JSON.stringify(quaertersums)}`);

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
            let summaryValue=JSON.parse(summaries[year][month].summary).summary;
            let summaryJSON=summaries[year][month].summary;
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
                    Summary__c: summaryJSON,
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
                    Summary__c: summaryJSON,
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
            tool_choice: "required",
            instructions: "You MUST call a function to process this request. Do NOT return text responses."
            //temperature: 0
        });
            
        console.log(`Run started: ${run.id}`);
        //console.log(`Run JSON: ${JSON.stringify(run)}`);
        const messages = await openai.beta.threads.messages.list(thread.id, {
            run_id: run.id,
          });
          // Log the full response structure
          //console.log(`OpenAI msg content Response: ${JSON.stringify(messages, null, 2)}`);
        if (!run.required_action?.submit_tool_outputs?.tool_calls) {
          throw new Error("Function call was expected but did not occur.");
        }
        console.log(`tool_calls: ${JSON.stringify(run.required_action.submit_tool_outputs.tool_calls)}`);
        console.log(`tool_calls arg: ${JSON.stringify(run.required_action.submit_tool_outputs.tool_calls[0].function.arguments)}`);

        const summaryObj = JSON.parse(run.required_action.submit_tool_outputs.tool_calls[0].function.arguments);
        // console.log("summaryObj :", summaryObj);
        // console.log("Type of summaryObj:", typeof summaryObj);
        // console.log("Keys in summaryObj:", Object.keys(summaryObj));
        // const summary = summaryObj.summary;
        // console.log("summary :", summary);
        // console.log("ActivityCount :", summaryObj.activityCount);
        // console.log("activityMapping :", summaryObj.activityMapping);

        const file = await openai.files.del(fileId);

          console.log(file);

        return summaryObj;
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
                Subject: activity.Subject || "No Subject",
                ActivityDate: activity.ActivityDate || "No Activity Date"
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

function transformStructure(oldJson) {
    const result = {};
    
    oldJson.yearlySummary.forEach(yearData => {
      const year = yearData.year;
      result[year] = {};
      
      yearData.quarters.forEach(quarterData => {
        result[year][quarterData.quarter] = {
          summary: JSON.stringify(quarterData),
          count: quarterData.activityCount,
          startdate: quarterData.startdate
        };
      });
    });
    
    return result;
  }


