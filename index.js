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
 * - Added detailed logging for CTA html_report generation.
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

// --- Enhanced Assistant Instructions ---
const OPENAI_MONTHLY_ASSISTANT_INSTRUCTIONS = `You are an AI assistant. Your task is to analyze Salesforce data (which could be Activities or CTAs) for a single month. You will be provided with the data and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function.
**Crucially, if the schema includes an 'html_report' field (this is especially important for CTA summaries):**
You MUST generate comprehensive and well-formatted HTML report content.
This HTML report should:
1.  Start with an \`<h1>\` tag containing the title (e.g., 'Monthly CTA Report for January 2024').
2.  Have distinct sections for each major insight category defined in the JSON output (e.g., SLA Analysis, Product Insights, Score/Grade Insights, Common Rejection Reasons, Description Insights, Action List). Use \`<h2>\` or \`<h3>\` tags for section headers.
3.  Present data clearly within these sections, using \`<ul>\` for lists, \`<p>\` for narrative text, and \`<table>\` for tabular data if appropriate (e.g., for 'conversion_vs_rejection_metrics').
4.  Ensure all key figures and findings from the JSON parameters are accurately reflected in the HTML. For example, if 'total_ctas' is 50, the HTML should state this. If 'sla_analysis.action' is 'Investigate delays', this action should be in the HTML.
5.  The 'action_list' array from the JSON should be rendered as an unordered list (\`<ul><li>...</li></ul>\`) in the HTML.
6.  The HTML should be self-contained and well-formed.
If data is provided as a file, use the file_search tool to access and process its content.
Do not include any explanations outside of the function call. Only provide the JSON.`;

const OPENAI_QUARTERLY_ASSISTANT_INSTRUCTIONS = `You are an AI assistant. Your task is to aggregate pre-summarized monthly Salesforce data (which could be from Activities or CTAs) into a consolidated quarterly summary. You will be provided with an array of monthly JSON summaries (these monthly summaries for CTAs will *not* include their original 'html_report'; you must generate a new one for the quarter) and a specific function schema (tailored for either activities or CTAs) during each run. You MUST generate a structured JSON output that strictly adheres to the provided function schema by calling that function. This includes correctly processing the input 'monthly_summaries' array and generating 'aggregated_data'.
**Crucially, if the schema includes an 'html_report' field (this is especially important for CTA summaries):**
You MUST generate a comprehensive and well-formatted HTML report content for the quarter based on your aggregated analysis of the 'monthly_summaries' input.
This quarterly HTML report should:
1.  Start with an \`<h1>\` tag containing the title (e.g., 'Quarterly CTA Report for Q1 2024').
2.  Synthesize and aggregate findings from the 'monthly_summaries' to populate the 'aggregated_data' JSON structure.
3.  Then, generate an HTML report reflecting this 'aggregated_data'.
4.  Have distinct sections for each major insight category from the 'aggregated_data' (e.g., Aggregated SLA Analysis, Consolidated Product Insights, Quarterly Score/Grade Trends, Common Rejection Reasons, Description Insights, Action List). Use \`<h2>\` or \`<h3>\` tags for section headers.
5.  Present aggregated data clearly, using \`<ul>\` for lists, \`<p>\` for narrative text, and \`<table>\` for tabular data if appropriate.
6.  Ensure all key figures and findings from the 'aggregated_data' JSON are accurately reflected in the HTML.
7.  The 'action_list' array from 'aggregated_data' should be rendered as an unordered list (\`<ul><li>...</li></ul>\`) in the HTML.
8.  The HTML should be self-contained and well-formed.
Do not include any explanations outside of the function call. Only provide the JSON.`;


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
                    "description": "The complete HTML summary report for the month. MUST be well-formed HTML. Start with an H1 title like 'Monthly CTA Report for {Month} {Year}'. Include H2/H3 sections for: Key Metrics (total, converted, rejected CTAs), SLA Analysis, Product Insights, Score/Grade Insights, Conversion vs. Rejection Metrics (consider using a simple table for clarity), Score and Product Breakdown, Common Rejected Reasons, Description Insights, and the Action List (as a UL). Ensure all data from other JSON parameters is represented in the HTML."
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
                    "description": "Array of structured monthly CTA summary inputs (output from 'generate_monthly_cta_summary' calls, EXCLUDING their 'html_report' field).",
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
                    "description": "The complete HTML summary report for the quarter, generated based on the aggregated_data. MUST be well-formed HTML. Start with an H1 title like 'Quarterly CTA Report for {Quarter} {Year}'. Include H2/H3 sections reflecting the aggregated_data for: Overall CTA Performance, SLA Analysis, Product Insights, Score/Grade Insights, Conversion vs. Rejection Metrics (consider a table), Score and Product Breakdown, Common Rejected Reasons, Description Insights, and the Action List (as a UL). Synthesize findings from the monthly inputs into this quarterly view."
                }
            },
            "required": ["quarter", "monthly_summaries", "aggregated_data", "html_report"],
            "definitions": { // Definitions remain the same
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
    assistantInstructions, // Using the passed instructions
    assistantToolsConfig,
    assistantModel
) {
    if (assistantIdEnvVar) {
        console.log(`Attempting to retrieve Assistant "${assistantName}" using ID: ${assistantIdEnvVar}`);
        try {
            const retrievedAssistant = await openaiClient.beta.assistants.retrieve(assistantIdEnvVar);
            // Optionally update the assistant if instructions or model changed significantly
            // For simplicity, we'll assume retrieval is sufficient if found
            // To update: await openaiClient.beta.assistants.update(assistantIdEnvVar, { instructions: assistantInstructions, tools: assistantToolsConfig, model: assistantModel });
            console.log(`Successfully retrieved existing Assistant "${retrievedAssistant.name}" with ID: ${retrievedAssistant.id}`);
            // If you want to ensure the assistant always has the latest instructions/tools from code:
            // console.log(`Updating Assistant "${assistantName}" with current configuration...`);
            // await openaiClient.beta.assistants.update(retrievedAssistant.id, {
            //     name: assistantName, // ensure name is consistent if it can change
            //     instructions: assistantInstructions,
            //     tools: assistantToolsConfig,
            //     model: assistantModel,
            // });
            // console.log(`Assistant "${assistantName}" updated.`);
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
            instructions: assistantInstructions, // Using the passed instructions
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
            OPENAI_MONTHLY_ASSISTANT_INSTRUCTIONS, // Pass defined instructions
            assistantBaseTools, OPENAI_MODEL
        );

        quarterlyAssistantId = await createOrRetrieveAssistant(
            openai, OPENAI_QUARTERLY_ASSISTANT_ID_ENV,
            "Salesforce Quarterly Data Aggregator",
            OPENAI_QUARTERLY_ASSISTANT_INSTRUCTIONS, // Pass defined instructions
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
                        records, openai, finalMonthlyAssistantId,
                        userPromptMonthly, finalMonthlyFuncSchema, summarObj
                    );

                    if (summarObj === "CTA") {
                        console.log(`[${accountId}] RAW Monthly CTA AI Output for ${month} ${year}:`, JSON.stringify(monthlySummaryResult, null, 2));
                        if (monthlySummaryResult && typeof monthlySummaryResult === 'object') {
                            console.log(`[${accountId}] Monthly CTA html_report for ${month} ${year} (type: ${typeof monthlySummaryResult.html_report}, length: ${monthlySummaryResult.html_report?.length || 0}): `, monthlySummaryResult.html_report ? `'${String(monthlySummaryResult.html_report).substring(0, 200)}...'` : 'MISSING or EMPTY');
                        } else {
                            console.log(`[${accountId}] Monthly CTA AI Output for ${month} ${year} is not a valid object or is null/undefined.`);
                        }
                    }

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
                 let html = '';
                 if (summarObj === "Activity") {
                    html = mData.aiOutput?.summary || '';
                 } else if (summarObj === "CTA") {
                    html = mData.aiOutput?.html_report || '';
                    console.log(`[${accountId}] Extracted aiSummaryHtml for CTA ${month} ${year} (to be saved, length: ${html?.length || 0}): `, html ? `'${String(html).substring(0,100)}...'` : 'EMPTY');
                 }
                 monthlyForSalesforce[year][month] = {
                     summary: JSON.stringify(mData.aiOutput), 
                     summaryDetails: html, 
                     count: mData.count, startdate: mData.startdate
                 };
             }
        }
        if (Object.values(monthlyForSalesforce).some(y => Object.keys(y).length > 0)) {
            console.log(`[${accountId}] Saving monthly ${summarObj} summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, monthlyForSalesforce, accountId, 'Monthly', summaryRecordsMap, loggedinUserId, summarObj);
            console.log(`[${accountId}] Monthly ${summarObj} summaries saved.`);
        } else {
            console.log(`[${accountId}] No monthly ${summarObj} summaries generated to save.`);
        }


        const quarterlyInputGroups = {};
        for (const year in finalMonthlySummaries) {
            for (const month in finalMonthlySummaries[year]) {
                const mData = finalMonthlySummaries[year][month];
                const qKey = `${year}-${getQuarterFromMonthIndex(mData.monthIndex)}`;
                if (!quarterlyInputGroups[qKey]) quarterlyInputGroups[qKey] = [];
                let aiOut = mData.aiOutput;
                if (summarObj === "CTA" && aiOut && typeof aiOut === 'object') {
                    const { html_report, ...rest } = aiOut; 
                    aiOut = rest; // Pass only the JSON data, not the monthly HTML report
                }
                quarterlyInputGroups[qKey].push(aiOut);
            }
        }

        const allQuarterlyRawResults = {};
        for (const [qKey, monthSummaries] of Object.entries(quarterlyInputGroups)) {
            if (!monthSummaries || monthSummaries.length === 0 || monthSummaries.every(s => !s)) { // check if all summaries are null/undefined
                console.warn(`[${accountId}] Skipping ${qKey} for ${summarObj} as it has no valid monthly summaries for quarterly aggregation.`);
                continue;
            }
            const [year, qtr] = qKey.split('-');
            // Ensure monthSummaries is not an array of nulls if some months had no data/errors previously
            const validMonthSummaries = monthSummaries.filter(s => s && typeof s === 'object' && Object.keys(s).length > 0);
            if (validMonthSummaries.length === 0) {
                console.warn(`[${accountId}] Skipping ${qKey} for ${summarObj} as all monthly summaries in the group are empty or invalid.`);
                continue;
            }

            const promptQtr = `${userPromptQuarterlyTemplate.replace('{{Quarter}}',qtr).replace('{{Year}}',year)}\n\nAggregate the following monthly ${summarObj} summary data for ${qKey} (note: for CTAs, monthly 'html_report' fields have been excluded; you must generate a new quarterly HTML report based on the aggregated data):\n\`\`\`json\n${JSON.stringify(validMonthSummaries,null,2)}\n\`\`\``;
            try {
                 const quarterlySummaryResult = await generateSummary(
                    null, openai, finalQuarterlyAssistantId, promptQtr, finalQuarterlyFuncSchema, summarObj
                 );
                 if (summarObj === "CTA") {
                    console.log(`[${accountId}] RAW Quarterly CTA AI Output for ${qKey}:`, JSON.stringify(quarterlySummaryResult, null, 2));
                    if (quarterlySummaryResult && typeof quarterlySummaryResult === 'object') {
                        console.log(`[${accountId}] Quarterly CTA html_report for ${qKey} (type: ${typeof quarterlySummaryResult.html_report}, length: ${quarterlySummaryResult.html_report?.length || 0}): `, quarterlySummaryResult.html_report ? `'${String(quarterlySummaryResult.html_report).substring(0, 200)}...'` : 'MISSING or EMPTY');
                    } else {
                        console.log(`[${accountId}] Quarterly CTA AI Output for ${qKey} is not a valid object or is null/undefined.`);
                    }
                 }
                 allQuarterlyRawResults[qKey] = quarterlySummaryResult;
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
            const totalQRecords = Object.values(finalQuarterlyDataForSalesforce).reduce((s, y) => s + Object.keys(y).length, 0);
            console.log(`[${accountId}] Saving ${totalQRecords} quarterly ${summarObj} summaries to Salesforce...`);
            await createTimileSummarySalesforceRecords(conn, finalQuarterlyDataForSalesforce, accountId, 'Quarterly', summaryRecordsMap, loggedinUserId, summarObj);
            console.log(`[${accountId}] Quarterly ${summarObj} summaries saved.`);
        } else {
            console.log(`[${accountId}] No quarterly ${summarObj} summaries transformed to save.`);
        }

        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Success", `${summarObj} Summary Processed Successfully`);
    } catch (error) {
        console.error(`[${accountId}] Error in ${summarObj} processSummary:`, error);
        await sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, "Failed", `${summarObj} Processing Error: ${error.message}`);
    }
}

async function generateSummary(
    records, openaiClient, assistantId, userPrompt, functionSchema, summarObj
) {
    let fileId = null, thread = null, filePath = null, inputMethod = "prompt";
    try {
        await fs.ensureDir(TEMP_FILE_DIR);
        thread = await openaiClient.beta.threads.create();
        console.log(`[Th ${thread.id}] Asst ${assistantId} (Task: ${summarObj}, Func: ${functionSchema.name})`);

        let finalUserPrompt = userPrompt, attachments = [];
        if (records?.length > 0) {
            let recJson = "";
            try {
                recJson = JSON.stringify(records, null, 2);
            } catch (stringifyError) {
                console.error(`[Th ${thread.id}] Error stringifying records for ${summarObj}:`, stringifyError);
                throw new Error(`Failed to stringify ${summarObj} data for processing.`);
            }

            let potentialPrompt = `${userPrompt}\n\nHere is the ${summarObj} data to process:\n\`\`\`json\n${recJson}\n\`\`\``;
            if (potentialPrompt.length < PROMPT_LENGTH_THRESHOLD && records.length <= DIRECT_INPUT_THRESHOLD) {
                inputMethod = "direct JSON"; finalUserPrompt = potentialPrompt;
            } else {
                inputMethod = "file upload";
                let recText = records.map((r, i) => `${summarObj} ${i+1}:\n`+Object.entries(r).map(([k,v])=>`  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join('\n')).join('\n\n---\n\n');
                filePath = path.join(TEMP_FILE_DIR, `sf_${summarObj.toLowerCase()}_${Date.now()}_${thread.id}.txt`);
                await fs.writeFile(filePath, recText);
                console.log(`[Th ${thread.id}] Temp file for ${summarObj} generated: ${filePath}`);
                const upRes = await openaiClient.files.create({ file: fs.createReadStream(filePath), purpose: "assistants" });
                fileId = upRes.id;
                console.log(`[Th ${thread.id}] File for ${summarObj} uploaded: ${fileId}`);
                attachments.push({ file_id: fileId, tools: [{ type: "file_search" }] });
                // For file upload, the prompt should refer to the file.
                // The assistant instructions already mention using file_search if data is in a file.
                // The userPrompt might need to be adjusted slightly if it assumes inline data.
                // For now, we'll assume the existing userPrompt is generic enough or refers to "the provided data".
            }
        }
        console.log(`[Th ${thread.id}] Using ${inputMethod} for ${summarObj}. Prompt length: ${finalUserPrompt.length}`);

        const msgPayload = { role: "user", content: finalUserPrompt };
        if (attachments.length > 0) msgPayload.attachments = attachments;
        await openaiClient.beta.threads.messages.create(thread.id, msgPayload);
        console.log(`[Th ${thread.id}] Message added for ${summarObj}.`);

        const run = await openaiClient.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistantId,
            tools: [{ type: "function", function: functionSchema }],
            tool_choice: { type: "function", function: { name: functionSchema.name } },
        });
        console.log(`[Th ${thread.id}] Run status for ${summarObj}: ${run.status}`);

        if (run.status === 'requires_action') {
            const toolCall = run.required_action?.submit_tool_outputs?.tool_calls?.[0];
            if (!toolCall || toolCall.type !== 'function' || toolCall.function.name !== functionSchema.name) {
                  throw new Error(`Unexpected tool call: Expected ${functionSchema.name}, Got ${toolCall?.function?.name || toolCall?.type}`);
            }
            const rawArgs = toolCall.function.arguments;
            console.log(`[Th ${thread.id}] Function call arguments received for ${summarObj} (${toolCall.function.name}). Length: ${rawArgs?.length}`);
            try {
                 const parsedSummaryObject = JSON.parse(rawArgs);
                 console.log(`[Th ${thread.id}] Successfully parsed function arguments for ${summarObj}.`);
                 // Extra check for CTA html_report
                 if (summarObj === "CTA" && functionSchema.name.includes("monthly")) {
                    console.log(`[Th ${thread.id}] Monthly CTA HTML report in parsed args (length ${parsedSummaryObject?.html_report?.length || 0}): '${String(parsedSummaryObject?.html_report).substring(0,100)}...'`);
                 } else if (summarObj === "CTA" && functionSchema.name.includes("quarterly")) {
                    console.log(`[Th ${thread.id}] Quarterly CTA HTML report in parsed args (length ${parsedSummaryObject?.html_report?.length || 0}): '${String(parsedSummaryObject?.html_report).substring(0,100)}...'`);
                 }
                 return parsedSummaryObject;
            } catch (parseError) {
                 console.error(`[Th ${thread.id}] Failed to parse args JSON for ${summarObj}:`, parseError, `Raw (first 500): ${rawArgs?.substring(0,500)}`);
                 throw new Error(`Failed to parse AI args for ${summarObj}: ${parseError.message}`);
            }
        } else if (run.status === 'completed') {
             const msgs = await openaiClient.beta.threads.messages.list(run.thread_id, {limit:1});
             const lastMessageContent = msgs.data[0]?.content[0]?.type === 'text' ? msgs.data[0].content[0].text.value : "N/A";
             console.warn(`[Th ${thread.id}] Run for ${summarObj} completed without func call. Last msg: ${lastMessageContent}`);
             throw new Error(`Assistant run for ${summarObj} completed without making the required function call to ${functionSchema.name}. Last message: ${lastMessageContent.substring(0, 200)}...`);
        } else {
             console.error(`[Th ${thread.id}] Run for ${summarObj} failed. Status: ${run.status}`, run.last_error);
             throw new Error(`Assistant run for ${summarObj} failed. Status: ${run.status}. Error: ${run.last_error?.message || 'Unknown'}`);
        }
    } catch (error) {
        console.error(`[Th ${thread?.id || 'N/A'}] Error in generateSummary for ${summarObj}: ${error.message}`, error.stack);
        throw error; 
    } finally {
        if (filePath) try { await fs.unlink(filePath); console.log(`[Th ${thread?.id}] Deleted temp file: ${filePath}`); } catch (e) { console.error(`Err del temp ${filePath}:`,e); }
        if (fileId) try { await openaiClient.files.del(fileId); console.log(`[Th ${thread?.id}] Deleted OpenAI file: ${fileId}`); } catch (e) { if (!(e instanceof NotFoundError || e?.status === 404)) console.error(`Err del OpenAI file ${fileId}:`,e); }
        // Deleting threads is good practice if not needed for later inspection.
        // if (thread) try { await openaiClient.beta.threads.del(thread.id); console.log(`[Th ${thread.id}] Deleted OpenAI thread.`); } catch(e) { console.error(`Err del OpenAI thread ${thread.id}:`, e); }
    }
}

async function createTimileSummarySalesforceRecords(conn, summaries, parentId, summaryCategory, summaryRecordsMap, loggedinUserId, summarObj) {
    console.log(`[${parentId}] Preparing to save ${summaryCategory} ${summarObj} summaries...`);
    let toCreate = [], toUpdate = [];
    for (const year in summaries) {
        for (const periodKey in summaries[year]) {
            const sData = summaries[year][periodKey];
            let jsonStr = sData.summary; 
            let html = sData.summaryDetails; 
            let startDate = sData.startdate;
            let count = sData.count;

            if (!html && jsonStr) {
                console.warn(`[${parentId}] Fallback: summaryDetails was empty for ${periodKey} ${year} (${summarObj}). Attempting re-parse from summary JSON.`);
                try {
                    const pJson = JSON.parse(jsonStr);
                    if (summarObj === "Activity") {
                        html = pJson?.summary || '';
                    } else if (summarObj === "CTA") {
                        html = pJson?.html_report || ''; // Check html_report for CTAs
                    }
                    if(html) console.log(`[${parentId}] Fallback successful, found HTML for ${periodKey} ${year} (${summarObj}). Length: ${html.length}`);
                    else console.warn(`[${parentId}] Fallback failed to find HTML for ${periodKey} ${year} (${summarObj}) in summary JSON.`);
                } catch (e) {
                    console.error(`[${parentId}] Error in fallback JSON parse for ${periodKey} ${year} (${summarObj}):`, e);
                 }
            }
            if (summarObj === "CTA") {
                console.log(`[${parentId}] Final HTML for ${summaryCategory} CTA ${periodKey} ${year} (length ${html?.length || 0}) before SF save: '${String(html).substring(0,100)}...'`);
            }


            let qtrVal = (summaryCategory === 'Quarterly') ? periodKey : null;
            let monthVal = (summaryCategory === 'Monthly') ? periodKey : null;
            let mapKey = (summaryCategory === 'Quarterly') ? `${qtrVal} ${year}` : `${monthVal.substring(0,3)} ${year}`;
            let existId = getValueByKey(summaryRecordsMap, mapKey);

            const payload = {
                Parent_Id__c: parentId, Month__c: monthVal, Year__c: String(year), Summary_Category__c: summaryCategory,
                Requested_By__c: loggedinUserId, Summary__c: jsonStr ? jsonStr.substring(0, 131072) : null,
                Summary_Details__c: html ? html.substring(0, 131072) : null, 
                FY_Quarter__c: qtrVal, Month_Date__c: startDate, Number_of_Records__c: count || 0, Type__c: summarObj,
            };

            if (!payload.Month_Date__c) { 
                console.warn(`[${parentId}] Skipping ${summarObj} record for ${mapKey} due to missing Month_Date__c.`);
                continue;
            }

            if (existId) {
                console.log(`[${parentId}]   Queueing update for ${summarObj} ${mapKey} (ID: ${existId})`);
                toUpdate.push({ Id: existId, ...payload });
            } else {
                console.log(`[${parentId}]   Queueing create for ${summarObj} ${mapKey}`);
                toCreate.push(payload);
            }
        }
    }
    const opts = { allOrNone: false };
    try {
        if (toCreate.length > 0) {
            console.log(`[${parentId}] Creating ${toCreate.length} new ${summaryCategory} ${summarObj} records...`);
            const res = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "insert", opts, toCreate);
            handleBulkResults(res, toCreate, 'create', parentId, summarObj);
        } else {
            console.log(`[${parentId}] No new ${summaryCategory} ${summarObj} records to create.`);
        }
        if (toUpdate.length > 0) {
            console.log(`[${parentId}] Updating ${toUpdate.length} existing ${summaryCategory} ${summarObj} records...`);
            const res = await conn.bulk.load(TIMELINE_SUMMARY_OBJECT_API_NAME, "update", opts, toUpdate);
            handleBulkResults(res, toUpdate, 'update', parentId, summarObj);
        } else {
            console.log(`[${parentId}] No existing ${summaryCategory} ${summarObj} records to update.`);
        }
    } catch (err) {
        console.error(`[${parentId}] Failed to save ${summaryCategory} ${summarObj} records to Salesforce: ${err.message}`, err);
        throw new Error(`Salesforce save for ${summarObj} failed: ${err.message}`);
    }
}

function handleBulkResults(results, payloads, opType, parentId, summarObj) {
    let s=0, f=0;
    results.forEach((res, i) => {
        const p = payloads[i];
        const idLog = p.Id || `${p.Month__c || p.FY_Quarter__c || 'N/A'} ${p.Year__c || 'N/A'}`;
        if (!res.success) { f++; console.error(`[${parentId}] Error ${opType} ${summarObj} record ${idLog}: ${JSON.stringify(res.errors)}`); }
        else s++;
    });
    console.log(`[${parentId}] Bulk ${opType} for ${summarObj} summary: ${s} succeeded, ${f} failed.`);
}

async function fetchRecords(conn, queryOrUrl, summarObj, allRecords = [], first = true) {
    try {
        const logPfx = first ? `Initial Query for ${summarObj}` : `Fetching next batch for ${summarObj}`;
        console.log(`[SF Fetch] ${logPfx}: ${String(queryOrUrl).substring(0,150)}...`);
        const res = first ? await conn.query(queryOrUrl) : await conn.queryMore(queryOrUrl);
        const fetched = res.records?.length || 0;
        if (fetched > 0) allRecords.push(...res.records);
        console.log(`[SF Fetch] Got ${fetched} ${summarObj} records. Total so far: ${allRecords.length}. Done: ${res.done}`);
        if (!res.done && res.nextRecordsUrl) {
            await new Promise(r => setTimeout(r, 200)); 
            return fetchRecords(conn, res.nextRecordsUrl, summarObj, allRecords, false);
        }
        console.log(`[SF Fetch] Finished fetching ${summarObj}. Total records retrieved: ${allRecords.length}. Grouping...`);
        return groupRecordsByMonthYear(allRecords, summarObj);
    } catch (error) {
        console.error(`[SF Fetch] Error fetching Salesforce ${summarObj} data: ${error.message}`, error);
        throw error;
    }
}

function groupRecordsByMonthYear(records, summarObj) {
    const grouped = {};
    const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    records.forEach(r => {
        // Use ActivityDate for Activities if available, otherwise CreatedDate as fallback.
        // For CTAs, CreatedDate seems to be the primary date field mentioned.
        let dateFieldToUse = r.CreatedDate;
        if (summarObj === "Activity" && r.ActivityDate) {
            dateFieldToUse = r.ActivityDate;
        } else if (!r.CreatedDate && summarObj === "CTA") { // CTAs must have CreatedDate based on schema.
            console.warn(`Skipping CTA record (ID: ${r.Id || 'Unknown'}) due to missing CreatedDate.`);
            return;
        }
         if (!dateFieldToUse) {
            console.warn(`Skipping ${summarObj} record (ID: ${r.Id || 'Unknown'}) due to missing primary date field (CreatedDate/ActivityDate).`);
            return;
        }

        try {
            const d = new Date(dateFieldToUse);
            if (isNaN(d.getTime())) {
                console.warn(`Skipping ${summarObj} record (ID: ${r.Id || 'Unknown'}) due to invalid date: ${dateFieldToUse}`);
                return;
            }
            const y = d.getUTCFullYear(), mIdx = d.getUTCMonth(), mName = months[mIdx];
            if (!grouped[y]) grouped[y] = [];
            let mEntry = grouped[y].find(e => e[mName]);
            if (!mEntry) { mEntry = { [mName]: [] }; grouped[y].push(mEntry); }

            if (summarObj === "Activity") {
                // Ensure ActivityDate is in YYYY-MM-DD format for AI consistency if it's just a date
                let activityDateStr = r.ActivityDate;
                if (r.ActivityDate) {
                    try {
                        activityDateStr = new Date(r.ActivityDate).toISOString().split('T')[0];
                    } catch (dateErr) {
                        console.warn(`Invalid ActivityDate ${r.ActivityDate} for record ${r.Id}, using raw.`);
                        activityDateStr = r.ActivityDate;
                    }
                } else { // Fallback to CreatedDate if ActivityDate is missing
                     try {
                        activityDateStr = new Date(r.CreatedDate).toISOString().split('T')[0];
                    } catch (dateErr) {
                        console.warn(`Invalid CreatedDate (fallback) ${r.CreatedDate} for record ${r.Id}, using raw.`);
                        activityDateStr = r.CreatedDate;
                    }
                }
                mEntry[mName].push({ Id:r.Id, Description:r.Description || null, Subject:r.Subject || null, ActivityDate: activityDateStr });
            } else if (summarObj === "CTA") {
                mEntry[mName].push({ 
                    Id:r.Id, Name:r.Name || null, CampaignGroup:r.Campaign_Group__c || null, CampaignSubType:r.Campaign_Sub_Type__c || null,
                    CampaignType:r.Campaign_Type__c || null, Contact:r.Contact__c || null, 
                    CreatedDate: new Date(r.CreatedDate).toISOString(), // Standardize to ISO string for AI
                    CreatedDateCustom:r.Created_Date_Custom__c || null, CurrentInterestedProductScoreGrade:r.Current_Interested_Product_Score_Grade__c || null,
                    CustomerSelectedProduct:r.Customer_Selected_Product__c || null, CustomersPerceivedSLA:r.CustomersPerceivedSLA__c || null,
                    DateContacted:r.Date_Contacted__c ? new Date(r.Date_Contacted__c).toISOString() : null, 
                    DateEmailed:r.Date_Emailed__c ? new Date(r.Date_Emailed__c).toISOString() : null, 
                    Description:r.Description__c || null,
                    DispositionedDate:r.Dispositioned_Date__c ? new Date(r.Dispositioned_Date__c).toISOString() : null, 
                    LeadScoreAccountSecurity:r.Lead_Score_Account_Security__c || null,
                    LeadScoreContactCenterTwilioFlex:r.Lead_Score_Contact_Center_Twilio_Flex__c || null,
                    LeadScoreSMSMessaging:r.Lead_Score_SMS_Messaging__c || null, LeadScoreVoiceIVRSIPTrunking:r.Lead_Score_Voice_IVR_SIP_Trunking__c || null,
                    MQLStatus:r.MQL_Status__c || null, MQLType:r.MQL_Type__c || null, CurrentOwnerFullName:r.Current_Owner_Full_Name__c || null,
                    Opportunity:r.Opportunity__c || null, ProductCategory:r.Product_Category__c || null, ProductType:r.Product_Type__c || null,
                    QualifiedLeadSource:r.Qualified_Lead_Source__c || null, RejectedReason:r.Rejected_Reason__c || null, Title:r.Title__c || null
                });
            }
        } catch(e) {
             console.warn(`Skipping ${summarObj} record (ID: ${r.Id || 'Unknown'}) due to date processing error: ${e.message}. Date value: ${dateFieldToUse}`);
        }
    });
    console.log(`Finished grouping ${summarObj} records by year and month.`);
    return grouped;
}

async function sendCallbackResponse(accountId, callbackUrl, loggedinUserId, accessToken, status, message) {
    const logMsg = String(message).substring(0,500)+(String(message).length>500?'...':'');
    console.log(`[${accountId}] Sending callback to ${callbackUrl}. Status: ${status}, Msg: ${logMsg}`);
    try {
        await axios.post(callbackUrl, {
                accountId, loggedinUserId, status: "Completed", // Standardize outer status
                processResult: (status === "Success" || status === "Failed") ? status : "Failed", // Inner specific result
                message
            }, { headers: { "Content-Type": "application/json", "Authorization": `Bearer ${accessToken}` }, timeout: 30000 }
        );
        console.log(`[${accountId}] Callback sent successfully.`);
    } catch (error) {
        let em = `Failed to send callback to ${callbackUrl}. `;
        if (error.response) em += `Status: ${error.response.status}, Data: ${JSON.stringify(error.response.data)}, Message: ${error.message}`;
        else if (error.request) em += `No response received. ${error.message}`; else em += `Error: ${error.message}`;
        console.error(`[${accountId}] ${em}`);
    }
}

function getValueByKey(arr, key) {
    if (!arr?.length) return null;
    const searchKeyLower = key.toLowerCase();
    const rec = arr.find(i => i?.key?.toLowerCase() === searchKeyLower);
    return rec ? rec.value : null;
}

function transformQuarterlyStructure(qAiOutput, qKey, summarObj) {
    const res = {};
    const [yStr, qStr] = qKey.split('-');
    const year = parseInt(yStr);

    if (!qAiOutput || typeof qAiOutput !== 'object' || !year || !qStr) {
        console.warn(`[Transform] Invalid input for ${summarObj} quarterly transformation (qKey: ${qKey}):`, { qAiOutput });
        return res; 
    }

    let html = '', count = 0, startDate = `${year}-${getQuarterStartMonth(qStr)}-01`; 

    if (summarObj === "Activity") {
        const yearData = qAiOutput.yearlySummary?.[0];
        const qData = yearData?.quarters?.[0];
        if (qData && qData.quarter === qStr && yearData.year === year) {
            html = qData.summary || '';
            count = qData.activityCount || 0;
            startDate = qData.startdate || startDate;
        } else {
             console.warn(`[Transform] Mismatched Activity quarter data for ${qKey}. Using defaults. AI Output:`, qAiOutput);
        }
    } else if (summarObj === "CTA") {
        html = qAiOutput.html_report || '';
        count = qAiOutput.aggregated_data?.total_ctas || 0;
        if (html) console.log(`[Transform] CTA Quarterly html_report for ${qKey} (length: ${html.length}, to be saved): '${String(html).substring(0,100)}...'`);
        else console.warn(`[Transform] CTA Quarterly html_report for ${qKey} is MISSING/EMPTY in AI output.`);
    }

    if (!res[year]) res[year] = {};
    res[year][qStr] = {
        summaryDetails: html, 
        summary: JSON.stringify(qAiOutput), // CORRECTED: Renamed from summaryJson to summary
        count: count,
        startdate: startDate
    };
    return res;
}

function getQuarterStartMonth(q) {
    if (!q || typeof q !== 'string') {
        console.warn(`Invalid quarter identifier "${q}" to getQuarterStartMonth. Defaulting Q1.`);
        return '01';
    }
    switch (q.toUpperCase()) {
        case 'Q1': return '01'; case 'Q2': return '04';
        case 'Q3': return '07'; case 'Q4': return '10';
        default:
            console.warn(`Unrecognized quarter "${q}" to getQuarterStartMonth. Defaulting Q1.`);
            return '01';
    }
}
