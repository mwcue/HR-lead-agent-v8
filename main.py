# main.py
"""
HR & Regional B2B Lead Generation Agent

This script automates the process of identifying companies from two streams:
1. HR software/service companies (globally/nationally)
2. General B2B companies operating in New England

It analyzes their potential pain points in the context of HR conference sponsorship
and gathers contact information.

The system uses the CrewAI framework to organize multiple AI agents:
- 1 Research Agent: Finds sources for both streams and extracts companies.
- 2 Analysis Agents (HR & NE B2B): Analyze companies within their specific context.
- 2 Reviewer Agents (HR & NE B2B): Review/refine pain points for relevance.

The resulting leads (tagged by category) are saved to a CSV file.

Usage:
    python main.py

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for LLM access
    SERPER_API_KEY: Serper API key for web search functionality

    Optional variables can be found in the config.py file
"""

import time
import logging # Import logging here
from dotenv import load_dotenv

# Import our custom modules
from url_processor import perform_search
from company_extractor import extract_companies_from_url, analyze_company
# Using original write_to_csv for now; appending/cross-run dedupe is next step
from output_manager import write_to_csv
from config import Config
# Import task creators (analysis/review tasks created inside analyze_company now)
from tasks import create_search_tasks, create_extraction_task
from utils.logging_utils import get_logger, ErrorCollection

# Configure logging
Config.configure_logging()
# Use get_logger here AFTER configure_logging is called
logger = get_logger(__name__) # Ensure logger is fetched after configuration

# Load environment variables
load_dotenv()

# Import tools and initialize agents
error_collector = ErrorCollection()
try:
    # --- Tool Imports ---
    from tools.scraper_tools import generic_scraper_tool
    from tools.unified_email_finder import unified_email_finder_tool
    from tools.llm_tools import analyze_pain_points_tool
    from tools.search_tools import web_search_tool
    # --- Agent Initialization ---
    from agents import initialize_agents # This now initializes 5 agents

    # Validate critical tools
    if web_search_tool is None:
        error_collector.add("Tool Initialization",
                           ValueError("Web Search Tool failed to initialize"),
                           fatal=True)
    else:
        logger.info("Custom tools imported successfully.")

    # Validate configuration
    missing_keys = Config.validate()
    if missing_keys:
        error_collector.add("Configuration",
                           ValueError(f"Missing required environment variables: {', '.join(missing_keys)}"),
                           fatal=True)

    # Check API keys
    logger.info(f"OPENAI_API_KEY presence: {'Yes' if Config.OPENAI_API_KEY else 'No'}")
    logger.info(f"SERPER_API_KEY presence: {'Yes' if Config.SERPER_API_KEY else 'No'}")

except ImportError as e:
    error_collector.add("Module Import", e, fatal=True)
except Exception as e:
    error_collector.add("Initialization", e, fatal=True)

# Exit if any fatal errors occurred during setup
if error_collector.has_fatal_errors():
    logger.error(error_collector.get_summary())
    exit(1)

# Initialize tools dictionary
tools = {
    'web_search': web_search_tool,
    'generic_scraper': generic_scraper_tool,
    'email_finder': unified_email_finder_tool,
    'pain_point_analyzer': analyze_pain_points_tool
    # Note: Reviewer agents currently have no tools assigned
}

# --- Simple Classifier Function ---
# TODO: Enhance this classification logic if needed (e.g., LLM call, more keywords)
def classify_company(company_name: str, company_website: str, source_url: str) -> str:
    """
    Classifies a company as 'HR' or 'NE_B2B' based on simple heuristics.
    Currently uses source URL keywords as a basic approach.
    """
    logger.debug(f"Classifying '{company_name}' from source: {source_url}")
    source_lower = source_url.lower()
    # Keywords suggesting an HR-focused source
    hr_keywords = ['hr-', 'human-resources', 'recruiting', 'payroll', 'benefits',
                   'talent-acquisition', 'hrtech', 'hcm', 'applicant-tracking']

    if any(keyword in source_lower for keyword in hr_keywords):
        logger.debug(f"Classified '{company_name}' as HR based on source URL.")
        return "HR"
    else:
        # If source doesn't scream HR, assume it's from the NE B2B search pool
        # This relies on the search tasks finding appropriate NE B2B sources.
        logger.debug(f"Classified '{company_name}' as NE_B2B (default).")
        return "NE_B2B"
# --- End Classifier ---


# Main execution block
if __name__ == "__main__":
    logger.info("\nStarting Dual Stream Lead Generation Crew (HR & NE B2B)...")

    all_processed_companies = []
    # Keep track of websites processed in this specific run
    processed_websites_this_run = set()

    # Initialize agents (expects the function returning 5 agents)
    try:
        agents = initialize_agents(tools)
        logger.info(f"Agents initialized successfully: {list(agents.keys())}")
        # Basic check for expected agent keys
        expected_keys = {'research', 'hr_analysis', 'hr_reviewer', 'ne_b2b_analysis', 'ne_b2b_reviewer'}
        if not expected_keys.issubset(agents.keys()):
            raise ValueError(f"Expected agents not found. Got: {list(agents.keys())}")
    except Exception as e:
        error_collector.add("Agent Initialization", e, fatal=True)
        logger.error(error_collector.get_summary())
        exit(1)

    # Step 1: Find relevant URLs (using the single Research Agent for both streams)
    # create_search_tasks is already updated to generate queries for both
    search_tasks = create_search_tasks(agents)
    url_list = perform_search(agents, search_tasks) # perform_search uses agents['research']

    # Step 2: Process Each Found URL
    if not url_list:
        logger.warning("No URLs found by Research Agent. Skipping processing phase.")
    else:
        logger.info(f"\n--- Starting Processing Loop for {len(url_list)} URLs (Potential HR or NE B2B sources) ---")
        urls_to_process = url_list[:Config.MAX_URLS_TO_PROCESS]
        logger.info(f"Processing first {len(urls_to_process)} URLs as per MAX_URLS_TO_PROCESS.")

        for i, target_url in enumerate(urls_to_process):
            logger.info(f"\nProcessing URL {i+1}/{len(urls_to_process)}: {target_url}")

            # Step 2a: Extract Companies from URL (uses Research Agent)
            # create_extraction_task is general enough
            extraction_task = create_extraction_task(target_url, agents)
            extracted_company_data = extract_companies_from_url(target_url, agents, extraction_task)

            # Step 2b: Classify and Analyze Each Company
            if not extracted_company_data:
                logger.info(f"  No companies extracted from {target_url}.")
                continue

            logger.info(f"  Analyzing {len(extracted_company_data)} companies extracted from {target_url}...")
            for company_dict in extracted_company_data:
                company_name = company_dict.get('name')
                company_website = company_dict.get('website')

                # Basic validation
                if not company_name or not company_website:
                    logger.warning(f"    Skipping entry with missing name/website: {company_dict}")
                    continue

                # --- Intra-Run Duplicate Check Logic (No change needed) ---
                normalized_website = company_website.strip().lower()
                if normalized_website.startswith('www.'):
                   normalized_website = normalized_website[4:]
                if normalized_website.endswith('/'):
                    normalized_website = normalized_website[:-1]

                if normalized_website in processed_websites_this_run:
                    logger.info(f"    Skipping already processed website in this run: '{company_name}' ({company_website})")
                    continue
                # --- End Intra-Run Check ---

                # Skip generic names (No change needed)
                if company_name.lower() in Config.GENERIC_COMPANY_NAMES:
                    logger.info(f"    Skipping generic name: '{company_name}'")
                    continue

                processed_websites_this_run.add(normalized_website) # Add website BEFORE analysis

                # --- NEW: Classify the company ---
                category = classify_company(company_name, company_website, target_url)
                # --- ADDED: Detailed Classification Log ---
                logger.info(f"    Company: '{company_name}' | Website: {company_website} | Source: {target_url} | Classified as: {category}")
                # --- END ADDED ---
                logger.info(f"    Analyzing '{company_name}' ({company_website}) as Category: {category}")

                # --- CHANGE: Call analyze_company WITH agents dict and category ---
                # analyze_company now handles routing to the correct agents internally
                company_data = analyze_company(company_name, company_website, agents, category)

                # Add source URL (which was already stored in company_data by analyze_company indirectly via initial dict)
                # Ensure source_url is present; analyze_company might fail early
                if company_data: # Check if analyze_company returned data
                    company_data["source_url"] = target_url # Add source URL
                    all_processed_companies.append(company_data)
                else:
                    # Should not happen often due to error handling in analyze_company, but good practice
                    logger.error(f"    Analysis for '{company_name}' returned None or empty. Skipping append.")


                # Short delay (Keep reasonable)
                time.sleep(Config.API_RETRY_DELAY / 2) # Link delay to config if desired

            logger.info(f"Finished processing companies from URL: {target_url}")
        logger.info("--- Finished Processing Loop ---")

    # Step 3: Write Final CSV Output
    if all_processed_companies:
        successful_analyses = [
            c for c in all_processed_companies
            if isinstance(c, dict) and not c.get("pain_points", "").startswith("Analysis failed") and not c.get("pain_points", "").startswith("Analysis skipped")
        ]
        # --- ADDED: Pre-CSV Classification Summary ---
        hr_count = sum(1 for c in all_processed_companies if isinstance(c, dict) and c.get('category') == 'HR')
        ne_b2b_count = sum(1 for c in all_processed_companies if isinstance(c, dict) and c.get('category') == 'NE_B2B')
        unknown_count = len(all_processed_companies) - hr_count - ne_b2b_count
        logger.info(f"\n--- Pre-CSV Classification Summary ---")
        logger.info(f"Total Processed Entries: {len(all_processed_companies)}")
        logger.info(f"Categorized as HR: {hr_count}")
        logger.info(f"Categorized as NE_B2B: {ne_b2b_count}")
        logger.info(f"Unknown/Failed Category: {unknown_count}")
        logger.info(f"Successful Analyses: {len(successful_analyses)}")
        logger.info(f"------------------------------------\n")
        # --- END ADDED ---
        logger.info(f"\nWriting final results ({len(successful_analyses)} successful analyses out of {len(all_processed_companies)} attempts) to {Config.OUTPUT_PATH}...")
        # write_to_csv will need the category column added
        write_to_csv(all_processed_companies, Config.OUTPUT_PATH)
    else:
        logger.warning("\nNo company data processed in this run. Skipping CSV output.")

    # Print error summary if any non-fatal errors occurred
    if error_collector.has_errors():
        logger.warning("\n--- Error Summary ---")
        logger.warning(error_collector.get_summary())
        logger.warning("--- End Error Summary ---")

    logger.info("\n--- End of Execution ---")
