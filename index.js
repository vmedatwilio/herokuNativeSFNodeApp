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

// --- Assistant IDs are now PREFERRED, creation is fallback (TWO Generic Assistants) ---
const OPENAI_MONTHLY_ASSISTANT_ID_ENV = process.env.OPENAI_MONTHLY_ASSISTANT_ID;
const OPENAI_QUARTERLY_ASSISTANT_ID_ENV = process.env.OPENAI_QUARTERLY_ASSISTANT_ID;
// CTA-specific assistant ID env vars are no longer used.

const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const TIMELINE_SUMMARY_OBJECT_API_NAME = "Timeline_Summary__c";
const DIRECT_INPUT_THRESHOLD = 2000;
const PROMPT_LENGTH_THRESHOLD = 256000;
const TEMP_FILE_DIR = path.join(__dirname, 'temp_files');

// --- Environment Variable Validation (Essential Vars) ---
if (!SF_LOGIN_URL || !OPENAI_API_KEY) {
    console.error("FATAL ERROR: Missing required environment variables (SF_LOGIN_URL, OPENAI_API_KEY).");
    process.exit(1);
}

// --- Global Variables for Final Assistant IDs ---
let monthlyAssistantId = null;
let quarterlyAssistantId = null;
// ctaMonthlyAssistantId and ctaQuarterlyAssistantId are removed.


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


// --- OpenAI Client Initialization ---
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// --- Express Application Setup ---
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// --- Helper Function to Create or Retrieve Assistant ---
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
        // Construct a generic env var name suggestion based on the role (Monthly/Quarterly)
        const role = assistantName.includes("Monthly") ? "MONTHLY" : assistantName.includes("Quarterly") ? "QUARTERLY" : "GENERIC";
        const envVarNameSuggestion = `OPENAI_${role}_ASSISTANT_ID`; // Suggest using the main env var names
        console.warn(`--> IMPORTANT: Consider adding this ID to your .env file as ${envVarNameSuggestion}=${newAssistant.id} for future reuse if this assistant is intended for this role.`);
        return newAssistant.id;
    } catch (creationError) {
        console.error(`Error creating Assistant "${assistantName}":`, creationError);
        throw new Error(`Failed to create Assistant ${assistantName}: ${creationError.message}`);
    }
}


// --- Server Startup ---
(async () => {
    try {
        console.log("Initializing Assistants...");
        await fs.ensureDir(TEMP_FILE_DIR);

        const assistantBaseTools = [{ type: "file_search" }, { type: "function" }];

        // --- Setup Monthly Assistant (Generic for Activities & CTAs) ---
        monthlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_MONTHLY_ASSISTANT_ID_ENV, // Use the generic monthly ID env var
            "Salesforce Monthly Data Summarizer",
            "You are an AI assistant. Your task is to analyze Salesforce data (which could be Activities or CTAs) for a single month. You will be provided with the data and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function. If the schema includes an 'html_report' field, you must generate the HTML report content for it. If data is provided as a file, use the file_search tool to access and process its content.",
            assistantBaseTools,
            OPENAI_MODEL
        );

        // --- Setup Quarterly Assistant (Generic for Activities & CTAs) ---
        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai,
            OPENAI_QUARTERLY_ASSISTANT_ID_ENV, // Use the generic quarterly ID env var
            "Salesforce Quarterly Data Aggregator",
            "You are an AI assistant. Your task is to aggregate pre-summarized monthly Salesforce data (which could be from Activities or CTAs) into a consolidated quarterly summary. You will be provided with an array of monthly JSON summaries and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function. This includes correctly processing the input 'monthly_summaries' array and generating aggregated data. If the schema includes an 'html_report' field, you must generate the HTML report content for it.",
            assistantBaseTools,
            OPENAI_MODEL
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


// --- Main API Endpoint ---
app.post('/generatesummary', async (req, res) => {
    console.log("Received /generatesummary request");

    const {
        accountId, callbackUrl, userPrompt, userPromptQtr, queryText,
        summaryMap, loggedinUserId, qtrJSON, monthJSON,
        summarObj // "Activity" or "CTA"
    } = req.body;

    // --- Ensure Core Assistants are Ready ---
    if (!monthlyAssistantId || !quarterlyAssistantId) { // Simplified check
        console.error("Error: Core Assistants not initialized properly.");
        return res.status(500).json({ error: "Internal Server Error: Core Assistants not ready." });
    }

    if (summarObj !== "Activity" && summarObj !== "CTA") {
        console.warn("Bad Request: Invalid 'summarObj' parameter. Must be 'Activity' or 'CTA'.");
        return res.status(400).send({ error: "Invalid 'summarObj' parameter. Must be 'Activity' or 'CTA'." });
    }

    const authHeader = req.headers["authorization"];
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
        console.warn("Unauthorized request: Missing or invalid Bearer token.");
        return res.status(401).json({ error: "Unauthorized" });
    }
    const accessToken = authHeader.split(" ")[1];

    if (!accountId || !callbackUrl || !accessToken || !queryText || !userPrompt || !userPromptQtr || !loggedinUserId || !summarObj) {
        console.warn("Bad Request: Missing required parameters.");
        return res.status(400).send({ error: "Missing required parameters (accountId, callbackUrl, accessToken, queryText, userPrompt, userPromptQtr, loggedinUserId, summarObj)" });
    }

    let summaryRecordsMap = {};
    let finalMonthlyFuncSchema, finalQuarterlyFuncSchema;
    // Assistant IDs are now fixed
    const currentMonthlyAssistantId = monthlyAssistantId;
    const currentQuarterlyAssistantId = quarterlyAssistantId;

    // Select function schemas based on summarObj
    if (summarObj === "Activity") {
        finalMonthlyFuncSchema = defaultActivityFunctions.find(f => f.name === 'generate_monthly_activity_summary');
        finalQuarterlyFuncSchema = defaultActivityFunctions.find(f => f.name === 'generate_quarterly_activity_summary');
    } else if (summarObj === "CTA") {
        finalMonthlyFuncSchema = defaultCtaFunctions.find(f => f.name === 'generate_monthly_cta_summary');
        finalQuarterlyFuncSchema = defaultCtaFunctions.find(f => f.name === 'generate_quarterly_cta_summary');
    }

    try {
        if (summaryMap) summaryRecordsMap = Object.entries(JSON.parse(summaryMap)).map(([key, value]) => ({ key, value }));
        if (monthJSON) {
            const customMonthSchema = JSON.parse(monthJSON);
            if (!customMonthSchema || typeof customMonthSchema !== 'object' || !customMonthSchema.name || !customMonthSchema.parameters) {
                throw new Error("Provided monthJSON schema is invalid.");
            }
            finalMonthlyFuncSchema = customMonthSchema;
            console.log(`Using custom monthly function schema for ${summarObj} from request.`);
        }
        if (qtrJSON) {
            const customQtrSchema = JSON.parse(qtrJSON);
            if (!customQtrSchema || typeof customQtrSchema !== 'object' || !customQtrSchema.name || !customQtrSchema.parameters) {
                throw new Error("Provided qtrJSON schema is invalid.");
            }
            finalQuarterlyFuncSchema = customQtrSchema;
            console.log(`Using custom quarterly function schema for ${summarObj} from request.`);
        }
    } catch (e) {
        console.error("Failed to parse JSON input from request body:", e);
        return res.status(400).send({ error: `Invalid JSON provided. ${e.message}` });
    }

    if (!finalMonthlyFuncSchema || !finalQuarterlyFuncSchema) {
        console.error(`FATAL: Function schemas for ${summarObj} could not be loaded.`);
        return res.status(500).send({ error: `Internal server error: Could not load function schemas for ${summarObj}.`});
    }

    res.status(202).json({ status: 'processing', message: `Summary generation for ${summarObj} initiated. You will receive a callback.` });
    console.log(`Initiating ${summarObj} summary processing for Account ID: ${accountId} using Monthly Asst: ${currentMonthlyAssistantId}, Qtrly Asst: ${currentQuarterlyAssistantId}`);

    processSummary(
        accountId, accessToken, callbackUrl, userPrompt, userPromptQtr, queryText,
        summaryRecordsMap, loggedinUserId,
        finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
        currentMonthlyAssistantId, currentQuarterlyAssistantId, // Pass the generic assistant IDs
        summarObj
    ).catch(async (error) => {
        console.error(`[${accountId}] Unhandled error during background processing for ${summarObj}:`, error);
        try {
            await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `Unhandled ${summarObj} processing error: ${error.message}`);
        } catch (callbackError) {
            console.error(`[${accountId}] Failed to send error callback for ${summarObj} after unhandled exception:`, callbackError);
        }
    });
});


// --- Helper Function to Get Quarter from Month Index ---
function getQuarterFromMonthIndex(monthIndex) {
    if (monthIndex >= 0 && monthIndex <= 2) return 'Q1';
    if (monthIndex >= 3 && monthIndex <= 5) return 'Q2';
    if (monthIndex >= 6 && monthIndex <= 8) return 'Q3';
    if (monthIndex >= 9 && monthIndex <= 11) return 'Q4';
    return 'Unknown';
}

// --- Asynchronous Summary Processing Logic ---
async function processSummary(
    accountId, accessToken, callbackUrl,
    userPromptMonthlyTemplate, userPromptQuarterlyTemplate, queryText,
    summaryRecordsMap, loggedinUserId,
    finalMonthlyFuncSchema, finalQuarterlyFuncSchema,
    finalMonthlyAssistantId, finalQuarterlyAssistantId, // These are now the generic IDs
    summarObj
) {
    console.log(`[${accountId}] Starting processSummary for ${summarObj} using Monthly Asst: ${finalMonthlyAssistantId}, Quarterly Asst: ${finalQuarterlyAssistantId}`);

    const conn = new jsforce.Connection({
        instanceUrl: SF_LOGIN_URL, accessToken: accessToken, maxRequest: 5, version: '59.0'
    });

    try {
        console.log(`[${accountId}] Fetching Salesforce records for ${summarObj}...`);
        const groupedData = await fetchRecords(conn, queryText, summarObj);
        const recordCount = Object.values(groupedData).flatMap(yearData => yearData.flatMap(monthObj => Object.values(monthObj)[0])).length;
        console.log(`[${accountId}] Fetched and grouped ${summarObj} data. Total records: ${recordCount}`);

        const finalMonthlySummaries = {};
        const monthMap = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };

        for (const year in groupedData) {
            console.log(`[${accountId}] Processing Year: ${year} for Monthly ${summarObj} Summaries`);
            finalMonthlySummaries[year] = {};
            for (const monthObj of groupedData[year]) {
                for (const month in monthObj) {
                    const records = monthObj[month];
                    console.log(`[${accountId}]   Processing ${summarObj} Month: ${month} (${records.length} records)`);
                    if (records.length === 0) {
                        console.log(`[${accountId}]   Skipping empty month for ${summarObj}: ${month} ${year}.`);
                        continue;
                    }
                    const monthIndex = monthMap[month.toLowerCase()];
                     if (monthIndex === undefined) {
                        console.warn(`[${accountId}]   Could not map month name: ${month}. Skipping ${summarObj} processing.`);
                        continue;
                    }
                    const startDate = new Date(Date.UTC(year, monthIndex, 1));
                    const userPromptMonthly = userPromptMonthlyTemplate.replace('{{YearMonth}}', `${month} ${year}`);

                    const monthlySummaryResult = await generateSummary(
                        records, openai, finalMonthlyAssistantId, // Use generic monthly assistant ID
                        userPromptMonthly, finalMonthlyFuncSchema, summarObj
                    );
                    finalMonthlySummaries[year][month] = {
                         aiOutput: monthlySummaryResult,
                         count: records.length,
                         startdate: startDate.toISOString().split('T')[0],
                         year: parseInt(year),
                         monthIndex: monthIndex
                    };
                    console.log(`[${accountId}]   Generated monthly ${summarObj} summary for ${month} ${year}.`);
                 }
            }
        }

        const monthlyForSalesforce = {};
        for (const year in finalMonthlySummaries) {
             monthlyForSalesforce[year] = {};
             for (const month in finalMonthlySummaries[year]) {
                 const monthData = finalMonthlySummaries[year][month];
                 let aiSummaryHtml = '';
                 if (summarObj === "Activity") {
                     aiSummaryHtml = monthData.aiOutput?.summary || '';
                 } else if (summarObj === "CTA") {
                     aiSummaryHtml = monthData.aiOutput?.html_report || '';
                 }

                 monthlyForSalesforce[year][month] = {
                     summary: JSON.stringify(monthData.aiOutput),
                     summaryDetails: aiSummaryHtml,
                     count: monthData.count,
                     startdate: monthData.startdate
                 };
             }
        }

        if (Object.keys(monthlyForSalesforce).length > 0 && Object.values(monthlyForSalesforce).some(yearData => Object.keys(yearData).length > 0)) {
            console.log(`[${accountId}] Saving monthly ${summarObj} summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap, loggedinUserId, summarObj);
            console.log(`[${accountId}] Monthly ${summarObj} summaries saved.`);
        } else {
             console.log(`[${accountId}] No monthly ${summarObj} summaries generated to save.`);
        }

        console.log(`[${accountId}] Grouping monthly ${summarObj} summaries by quarter...`);
        const quarterlyInputGroups = {};
        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const monthData = finalMonthlySummaries[year][month];
                const quarter = getQuarterFromMonthIndex(monthData.monthIndex);
                const quarterKey = `${year}-${quarter}`;
                if (!quarterlyInputGroups[quarterKey]) quarterlyInputGroups[quarterKey] = [];

                let monthlyAiOutputForQuarterly = monthData.aiOutput;
                if (summarObj === "CTA" && monthlyAiOutputForQuarterly) {
                    const { html_report, ...rest } = monthlyAiOutputForQuarterly;
                    monthlyAiOutputForQuarterly = rest;
                }
                quarterlyInputGroups[quarterKey].push(monthlyAiOutputForQuarterly);
            }
        }
        console.log(`[${accountId}] Identified ${Object.keys(quarterlyInputGroups).length} quarters with ${summarObj} data.`);

        const allQuarterlyRawResults = {};
        for (const [quarterKey, monthlySummariesForQuarter] of Object.entries(quarterlyInputGroups)) {
            console.log(`[${accountId}] Generating quarterly ${summarObj} summary for ${quarterKey} using ${monthlySummariesForQuarter.length} monthly summaries...`);
            if (!monthlySummariesForQuarter || monthlySummariesForQuarter.length === 0) {
                console.warn(`[${accountId}] Skipping ${quarterKey} for ${summarObj} as it has no monthly summaries.`);
                continue;
            }

            const quarterlyInputDataString = JSON.stringify(monthlySummariesForQuarter, null, 2);
            const [year, quarter] = quarterKey.split('-');
            const userPromptQuarterly = `${userPromptQuarterlyTemplate.replace('{{Quarter}}', quarter).replace('{{Year}}', year)}\n\nAggregate the following monthly ${summarObj} summary data for ${quarterKey}:\n\`\`\`json\n${quarterlyInputDataString}\n\`\`\``;

            try {
                 const quarterlySummaryResult = await generateSummary(
                    null, openai, finalQuarterlyAssistantId, // Use generic quarterly assistant ID
                    userPromptQuarterly, finalQuarterlyFuncSchema, summarObj
                 );
                 allQuarterlyRawResults[quarterKey] = quarterlySummaryResult;
                 console.log(`[${accountId}] Successfully generated quarterly ${summarObj} summary for ${quarterKey}.`);
            } catch (quarterlyError) {
                 console.error(`[${accountId}] Failed to generate quarterly ${summarObj} summary for ${quarterKey}:`, quarterlyError);
            }
        }

        console.log(`[${accountId}] Transforming ${Object.keys(allQuarterlyRawResults).length} generated quarterly ${summarObj} summaries...`);
        const finalQuarterlyDataForSalesforce = {};
        for (const [quarterKey, rawAiResult] of Object.entries(allQuarterlyRawResults)) {
             const transformedResult = transformQuarterlyStructure(rawAiResult, quarterKey, summarObj);
             for (const year in transformedResult) {
                 if (!finalQuarterlyDataForSalesforce[year]) finalQuarterlyDataForSalesforce[year] = {};
                 for (const quarter in transformedResult[year]) {
                     finalQuarterlyDataForSalesforce[year][quarter] = transformedResult[year][quarter];
                 }
             }
        }

        if (Object.keys(finalQuarterlyDataForSalesforce).length > 0 && Object.values(finalQuarterlyDataForSalesforce).some(yearData => Object.keys(yearData).length > 0)) {
            const totalQuarterlyRecords = Object.values(finalQuarterlyDataForSalesforce).reduce((sum, yearData) => sum + Object.keys(yearData).length, 0);
            console.log(`[${accountId}] Saving ${totalQuarterlyRecords} quarterly ${summarObj} summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId, summarObj);
            console.log(`[${accountId}] Quarterly ${summarObj} summaries saved.`);
        } else {
             console.log(`[${accountId}] No quarterly ${summarObj} summaries generated or transformed to save.`);
        }

        console.log(`[${accountId}] ${summarObj} Process completed.`);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", `${summarObj} Summary Processed Successfully`);

    } catch (error) {
        console.error(`[${accountId}] Error during ${summarObj} summary processing:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `${summarObj} Processing error: ${error.message}`);
    }
}


// --- OpenAI Summary Generation Function (assistantId is now generic) ---
async function generateSummary(
    records,
    openaiClient,
    assistantId, // This will be the generic monthly or quarterly ID
    userPrompt,
    functionSchema,
    summarObj
) {
    let fileId = null;
    let thread = null;
    let filePath = null;
    let inputMethod = "prompt";

    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        thread = await openaiClient.beta.threads.create();
        console.log(`[Thread ${thread.id}] Created for Assistant ${assistantId} (Task: ${summarObj}, Function: ${functionSchema.name})`);

        let finalUserPrompt = userPrompt;
        let messageAttachments = [];

        if (records && Array.isArray(records) && records.length > 0) {
            let recordsJsonString = JSON.stringify(records, null, 2);
            let potentialFullPrompt = `${userPrompt}\n\nHere is the ${summarObj} data to process:\n\`\`\`json\n${recordsJsonString}\n\`\`\``;
            console.log(`[Thread ${thread.id}] Potential prompt length for ${summarObj} with direct JSON: ${potentialFullPrompt.length} characters.`);

            if (potentialFullPrompt.length < PROMPT_LENGTH_THRESHOLD && records.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON";
                finalUserPrompt = potentialFullPrompt;
                console.log(`[Thread ${thread.id}] Using direct JSON input for ${summarObj}.`);
            } else {
                inputMethod = "file upload";
                console.log(`[Thread ${thread.id}] Using file upload for ${summarObj}.`);
                finalUserPrompt = userPrompt;

                let recordsText = records.map((record, index) => {
                    let recordLines = [`${summarObj} ${index + 1}:`];
                    for (const [key, value] of Object.entries(record)) {
                        let displayValue = value === null || value === undefined ? 'N/A' :
                                           typeof value === 'object' ? JSON.stringify(value) : String(value);
                        recordLines.push(`  ${key}: ${displayValue}`);
                    }
                    return recordLines.join('\n');
                }).join('\n\n---\n\n');

                const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
                const filename = `salesforce_${summarObj.toLowerCase()}_${timestamp}_${thread.id}.txt`;
                filePath = path.join(TEMP_FILE_DIR, filename);
                await fs.writeFile(filePath, recordsText);
                console.log(`[Thread ${thread.id}] Temporary text file for ${summarObj} generated: ${filePath}`);

                const uploadResponse = await openaiClient.files.create({
                    file: fs.createReadStream(filePath), purpose: "assistants",
                });
                fileId = uploadResponse.id;
                console.log(`[Thread ${thread.id}] File for ${summarObj} uploaded to OpenAI: ${fileId}`);
                messageAttachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                console.log(`[Thread ${thread.id}] Attaching file ${fileId} with file_search tool for ${summarObj}.`);
            }
        } else {
             console.log(`[Thread ${thread.id}] No ${summarObj} records array provided or array is empty. Using prompt content as is.`);
        }

        const messagePayload = { role: "user", content: finalUserPrompt };
        if (messageAttachments.length > 0) messagePayload.attachments = messageAttachments;
        const message = await openaiClient.beta.threads.messages.create(thread.id, messagePayload);
        console.log(`[Thread ${thread.id}] Message added for ${summarObj} (using ${inputMethod}). ID: ${message.id}`);

        console.log(`[Thread ${thread.id}] Starting run for ${summarObj}, Assistant: ${assistantId}, Forcing function: ${functionSchema.name}`);
        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }], // Provide the specific schema for this task
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Thread ${thread.id}] Run status for ${summarObj}: ${run.status}`);

        if (run.status === 'requires_action') {
            const toolCalls = run.required_action?.submit_tool_outputs?.tool_calls;
            if (!toolCalls || toolCalls.length === 0) {
                 throw new Error("Function call was expected but not provided by the Assistant.");
            }
            const toolCall = toolCalls[0];
            if (toolCall.type !== 'function' || toolCall.function.name !== functionSchema.name) {
                  throw new Error(`Assistant required action for unexpected tool: Expected ${functionSchema.name}, Got ${toolCall.function?.name || toolCall.type}`);
            }
            const rawArgs = toolCall.function.arguments;
            console.log(`[Thread ${thread.id}] Function call arguments received for ${summarObj} (${toolCall.function.name}).`);
            try {
                 const parsedSummaryObject = JSON.parse(rawArgs);
                 console.log(`[Thread ${thread.id}] Successfully parsed function arguments for ${summarObj}.`);
                 return parsedSummaryObject;
            } catch (parseError) {
                 console.error(`[Thread ${thread.id}] Failed to parse function call arguments JSON for ${summarObj}:`, parseError, `Raw args: ${rawArgs.substring(0,500)}...`);
                 throw new Error(`Failed to parse function call arguments from AI for ${summarObj}: ${parseError.message}`);
            }
        } else if (run.status === 'completed') {
             console.warn(`[Thread ${thread.id}] Run for ${summarObj} completed without requiring function call, despite tool_choice. Check Assistant setup/prompt.`);
             const messagesList = await openaiClient.beta.threads.messages.list(run.thread_id, { limit: 1 });
             throw new Error(`Assistant run for ${summarObj} completed without making the required function call. Last message: ${messagesList.data[0]?.content[0]?.text?.value || "N/A"}`);
        } else {
             console.error(`[Thread ${thread.id}] Run for ${summarObj} failed. Status: ${run.status}`, run.last_error);
             throw new Error(`Assistant run for ${summarObj} failed. Status: ${run.status}. Error: ${run.last_error ? run.last_error.message : 'Unknown'}`);
        }
    } catch (error) {
        console.error(`[Thread ${thread?.id || 'N/A'}] Error in generateSummary for ${summarObj}: ${error.message}`, error);
        throw error;
    } finally {
        if (filePath) {
            try { await fs.unlink(filePath); console.log(`[Thread ${thread?.id}] Deleted temp file: ${filePath}`); }
            catch (e) { console.error(`[Thread ${thread?.id}] Error deleting temp file ${filePath}:`, e); }
        }
        if (fileId) {
            try { await openaiClient.files.del(fileId); console.log(`[Thread ${thread?.id}] Deleted OpenAI file: ${fileId}`); }
            catch (e) { if (!(e instanceof NotFoundError || e?.status === 404)) console.error(`[Thread ${thread?.id}] Error deleting OpenAI file ${fileId}:`, e); }
        }
    }
}

// --- Salesforce Record Creation/Update Function ---
async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap, loggedinUserId, summarObj) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} ${summarObj} summaries...`);
    let recordsToCreate = [];
    let recordsToUpdate = [];

    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const summaryData = summaries[year][periodKey];
            let summaryJsonString = summaryData.summary;
            let summaryDetailsHtml = summaryData.summaryDetails;
            let startDate = summaryData.startdate;
            let count = summaryData.count;

            if (!summaryDetailsHtml && summaryJsonString) {
                try {
                    const parsedJson = JSON.parse(summaryJsonString);
                    if (summarObj === "Activity") {
                        summaryDetailsHtml = parsedJson?.summary || '';
                    } else if (summarObj === "CTA") {
                        summaryDetailsHtml = parsedJson?.html_report || '';
                    }
                } catch (e) {
                    console.warn(`[${parentId}] Could not parse 'summaryJsonString' for ${periodKey} ${year} (${summarObj}) to extract HTML details as fallback.`);
                }
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
                Summary__c: summaryJsonString ? summaryJsonString.substring(0, 131072) : null,
                Summary_Details__c: summaryDetailsHtml ? summaryDetailsHtml.substring(0, 131072) : null,
                FY_Quarter__c: fyQuarterValue || null,
                Month_Date__c: startDate,
                Number_of_Records__c: count || 0,
                Type__c: summarObj,
            };

            if (!recordPayload.Summary_Category__c || !recordPayload.Month_Date__c) {
                 console.warn(`[${parentId}] Skipping ${summarObj} record for ${summaryMapKey} due to missing Category or Start Date.`);
                 continue;
            }

            if (existingRecordId) {
                console.log(`[${parentId}]   Queueing update for ${summarObj} ${summaryMapKey} (ID: ${existingRecordId})`);
                recordsToUpdate.push({ Id: existingRecordId, ...recordPayload });
            } else {
                console.log(`[${parentId}]   Queueing create for ${summarObj} ${summaryMapKey}`);
                recordsToCreate.push(recordPayload);
            }
        }
    }

    try {
        const options = { allOrNone: false };
        if (recordsToCreate.length > 0) {
            console.log(`[${parentId}] Creating ${recordsToCreate.length} new ${summaryCategory} ${summarObj} summary records...`);
            const createResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", options, recordsToCreate);
            handleBulkResults(createResults, recordsToCreate, 'create', parentId, summarObj);
        }
        if (recordsToUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${recordsToUpdate.length} existing ${summaryCategory} ${summarObj} summary records...`);
             const updateResults = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", options, recordsToUpdate);
             handleBulkResults(updateResults, recordsToUpdate, 'update', parentId, summarObj);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} ${summarObj} records: ${err.message}`, err);
        throw new Error(`Salesforce save for ${summarObj} failed: ${err.message}`);
    }
}

// Helper to log bulk API results
function handleBulkResults(results, originalPayloads, operationType, parentId, summarObj) {
    console.log(`[${parentId}] Bulk ${operationType} for ${summarObj} results received (${results.length}).`);
    let successes = 0; let failures = 0;
    results.forEach((res, index) => {
        const p = originalPayloads[index];
        const idForLog = p.Id || `${p.Month__c || p.FY_Quarter__c || 'N/A'} ${p.Year__c || 'N/A'}`;
        if (!res.success) {
            failures++;
            console.error(`[${parentId}] Error ${operationType} ${summarObj} record ${index + 1} (${idForLog}): ${JSON.stringify(res.errors)}`);
        } else { successes++; }
    });
    console.log(`[${parentId}] Bulk ${operationType} for ${summarObj} summary: ${successes} succeeded, ${failures} failed.`);
}


// --- Salesforce Data Fetching with Pagination ---
async function fetchRecords(conn, queryOrUrl, summarObj, allRecords = [], isFirstIteration = true) {
    try {
        const logPrefix = isFirstIteration ? `Initial Query for ${summarObj}` : `Fetching next batch for ${summarObj}`;
        console.log(`[SF Fetch] ${logPrefix}: ${(queryOrUrl || '').substring(0,150)}...`);
        const queryResult = isFirstIteration ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);
        const fetchedCount = queryResult.records ? queryResult.records.length : 0;
        console.log(`[SF Fetch] Fetched ${fetchedCount} ${summarObj} records. Total so far: ${allRecords.length + fetchedCount}. Done: ${queryResult.done}`);
        if (fetchedCount > 0) allRecords = allRecords.concat(queryResult.records);

        if (!queryResult.done && queryResult.nextRecordsUrl) {
            await new Promise(resolve => setTimeout(resolve, 200));
            return fetchRecords(conn, queryResult.nextRecordsUrl, summarObj, allRecords, false);
        } else {
            console.log(`[SF Fetch] Finished fetching ${summarObj}. Total records: ${allRecords.length}. Grouping...`);
            return groupRecordsByMonthYear(allRecords, summarObj);
        }
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce ${summarObj} data: ${error.message}`, error);
        throw error;
    }
}


// --- Data Grouping Helper Function ---
function groupRecordsByMonthYear(records, summarObj) {
    const groupedData = {};
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

    records.forEach(record => {
        if (!record.CreatedDate) {
            console.warn(`Skipping ${summarObj} record (ID: ${record.Id || 'Unknown'}) due to missing CreatedDate.`);
            return;
        }
        try {
            const date = new Date(record.CreatedDate);
            if (isNaN(date.getTime())) {
                 console.warn(`Skipping ${summarObj} record (ID: ${record.Id || 'Unknown'}) due to invalid CreatedDate: ${record.CreatedDate}`);
                 return;
            }
            const year = date.getUTCFullYear();
            const monthIndex = date.getUTCMonth();
            const month = monthNames[monthIndex];

            if (!groupedData[year]) groupedData[year] = [];
            let monthEntry = groupedData[year].find(entry => entry[month]);
            if (!monthEntry) {
                monthEntry = { [month]: [] };
                groupedData[year].push(monthEntry);
            }

            if (summarObj === "Activity") {
                monthEntry[month].push({
                    Id: record.Id,
                    Description: record.Description || null,
                    Subject: record.Subject || null,
                    ActivityDate: record.CreatedDate // Using CreatedDate for grouping, ensure SOQL provides specific ActivityDate if needed by AI
                });
            } else if (summarObj === "CTA") {
                monthEntry[month].push({
                    Id: record.Id, Name: record.Name || null, CampaignGroup: record.Campaign_Group__c || null,
                    CampaignSubType: record.Campaign_Sub_Type__c || null, CampaignType: record.Campaign_Type__c || null,
                    Contact: record.Contact__c || null, CreatedDate: record.CreatedDate, // Important for AI context
                    CreatedDateCustom: record.Created_Date_Custom__c || null,
                    CurrentInterestedProductScoreGrade: record.Current_Interested_Product_Score_Grade__c || null,
                    CustomerSelectedProduct: record.Customer_Selected_Product__c || null,
                    CustomersPerceivedSLA: record.CustomersPerceivedSLA__c || null, DateContacted: record.Date_Contacted__c || null,
                    DateEmailed: record.Date_Emailed__c || null, Description: record.Description__c || null,
                    DispositionedDate: record.Dispositioned_Date__c || null,
                    LeadScoreAccountSecurity: record.Lead_Score_Account_Security__c || null,
                    LeadScoreContactCenterTwilioFlex: record.Lead_Score_Contact_Center_Twilio_Flex__c || null,
                    LeadScoreSMSMessaging: record.Lead_Score_SMS_Messaging__c || null,
                    LeadScoreVoiceIVRSIPTrunking: record.Lead_Score_Voice_IVR_SIP_Trunking__c || null,
                    MQLStatus: record.MQL_Status__c || null, MQLType: record.MQL_Type__c || null,
                    CurrentOwnerFullName: record.Current_Owner_Full_Name__c || null, Opportunity: record.Opportunity__c || null,
                    ProductCategory: record.Product_Category__c || null, ProductType: record.Product_Type__c || null,
                    QualifiedLeadSource: record.Qualified_Lead_Source__c || null, RejectedReason: record.Rejected_Reason__c || null,
                    Title: record.Title__c || null
                });
            }
        } catch(dateError) {
             console.warn(`Skipping ${summarObj} record (ID: ${record.Id || 'Unknown'}) due to date processing error: ${dateError.message}. Date: ${record.CreatedDate}`);
        }
    });
    console.log(`Finished grouping ${summarObj} records by year and month.`);
    return groupedData;
}


// --- Callback Sending Function ---
async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logMessage = message.length > 500 ? message.substring(0, 500) + '...' : message;
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Message: ${logMessage}`);
    try {
        const processStatus = (status === "Success" || status === "Failed") ? status : "Failed";
        await axios.post(callbackUrl, {
                accountId: accountId, loggedinUserId: loggedinUserId, status: "Completed",
                processResult: processStatus, message: message
            }, {
                headers: { "Content-Type": "application/json", "Authorization": `Bearer ${accessToken}` },
                timeout: 30000
            }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let em = `Failed to send callback to ${callbackUrl}. `;
        if (error.response) em += `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        else if (error.request) em += `No response received. ${error.message}`;
        else em += `Error: ${error.message}`;
        console.error(`[${accountId}] ${em}`);
    }
}


// --- Utility Helper Functions ---
function getValueByKey(recordsArray, searchKey) {
    if (!recordsArray || !Array.isArray(recordsArray)) return null;
    const record = recordsArray.find(item => item && typeof item.key === 'string' && item.key.toLowerCase() === searchKey.toLowerCase());
    return record ? record.value : null;
}

function transformQuarterlyStructure(quarterlyAiOutput, quarterKey, summarObj) {
    const result = {};
    const [yearStr, quarterStr] = quarterKey.split('-');
    const year = parseInt(yearStr);
    const quarter = quarterStr;

    if (!quarterlyAiOutput || typeof quarterlyAiOutput !== 'object' || !year || !quarter) {
        console.warn(`[Transform] Invalid input for ${summarObj} quarterly transformation:`, { quarterlyAiOutput, quarterKey });
        return result;
    }

    let htmlSummary = '';
    let fullQuarterlyJson = JSON.stringify(quarterlyAiOutput);
    let recordCount = 0;
    let startDate = `${year}-${getQuarterStartMonth(quarter)}-01`; // Default start date

    if (summarObj === "Activity") {
        const yearData = quarterlyAiOutput.yearlySummary?.[0];
        const quarterData = yearData?.quarters?.[0];
        if (quarterData && quarterData.quarter === quarter && yearData.year === year) {
            htmlSummary = quarterData.summary || '';
            recordCount = quarterData.activityCount || 0;
            startDate = quarterData.startdate || startDate; // Use AI's startdate if available
        } else {
            console.warn(`[Transform] Mismatched Activity quarter data for ${quarterKey}. Using defaults. AI Output:`, quarterlyAiOutput);
        }
    } else if (summarObj === "CTA") {
        htmlSummary = quarterlyAiOutput.html_report || '';
        recordCount = quarterlyAiOutput.aggregated_data?.total_ctas || 0;
        // For CTA, start_date is not part of its quarterly AI output schema, so we calculate it.
    }

    if (!result[year]) result[year] = {};
    result[year][quarter] = {
        summaryDetails: htmlSummary,
        summaryJson: fullQuarterlyJson,
        count: recordCount,
        startdate: startDate
    };
    return result;
}

function getQuarterStartMonth(quarter) {
    if (!quarter || typeof quarter !== 'string') {
        console.warn(`Invalid quarter identifier "${quarter}" to getQuarterStartMonth. Defaulting Q1.`);
        return '01';
    }
    switch (quarter.toUpperCase()) {
        case 'Q1': return '01'; case 'Q2': return '04';
        case 'Q3': return '07'; case 'Q4': return '10';
        default: console.warn(`Unrecognized quarter "${quarter}" to getQuarterStartMonth. Defaulting Q1.`); return '01';
    }
}
