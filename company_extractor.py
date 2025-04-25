# company_extractor.py
import logging
import ast
import re
import time
# Import CrewOutput and Crew, Agent, Task
from crewai import Crew, Process, CrewOutput, Agent, Task
from config import Config
# Import task creators
from tasks import create_analysis_task, create_review_task

# Configure logging
logger = logging.getLogger(__name__)

def parse_company_website_list(agent_output: str) -> list:
    """
    Extracts a list of {'name': ..., 'website': ...} dicts from agent output string.

    Args:
        agent_output: String output from agent containing company data

    Returns:
        List of dictionaries with company information
    """
    logger.debug(f"Parsing Company/Website list from raw agent output string.")
    company_data = []

    if not isinstance(agent_output, str):
        logger.warning(f"Company/Website Parser received non-string input: {type(agent_output)}. Cannot parse.")
        return company_data

    # Clean potential prefixes/suffixes before parsing
    if agent_output.strip().upper().startswith("FINAL ANSWER:"):
         agent_output = agent_output.split(":", 1)[1].strip()
    agent_output = agent_output.strip().strip('```python').strip('```').strip()

    try: # Try ast.literal_eval
        potential_list = ast.literal_eval(agent_output)
        if isinstance(potential_list, list):
            for item in potential_list:
                if isinstance(item, dict):
                    name = item.get('name')
                    website = item.get('website')
                    # Add robust check for website format
                    if isinstance(name, str) and isinstance(website, str) and name.strip() and website.strip().startswith('http') and 1 < len(name) < 60:
                        company_data.append({'name': name.strip(), 'website': website.strip()})
            if company_data: # Log only if something was parsed
                # NOTE: Temporarily changed log level to INFO for visibility during debug
                logger.info(f"Parsed Company/Website list via ast: {len(company_data)} entries.")
                return company_data
            else:
                logger.debug("ast.literal_eval resulted in a list, but no valid company entries found.")
        else:
            logger.warning(f"ast.literal_eval did not return a list: {type(potential_list)}")
    except Exception as e:
        logger.warning(f"Could not parse company/website list via ast: {e}")

    logger.warning("Failed to parse company/website list from agent output.")
    return [] # Return empty list if parsing fails

def parse_analysis_results(result: str) -> dict:
    """
    Parse the analysis results to extract email and pain points.

    Args:
        result: String output from analysis agent

    Returns:
        Dictionary with extracted email and pain points
    """
    logger.debug("Parsing analysis results")

    if not isinstance(result, str):
        logger.warning(f"Analysis result is not a string: {type(result)}. Cannot parse.")
        return {"email": "", "pain_points": "Analysis failed - non-string result"}

    # Remove "FINAL ANSWER:" prefix if present
    if result.strip().upper().startswith("FINAL ANSWER:"):
        result = result.split(":", 1)[1].strip()

    # Initialize default returns
    email = ""
    pain_points = ""

    # Look for email addresses in the text
    email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', result)
    valid_emails = [e for e in email_matches if '@' in e and '.' in e.split('@')[-1] and not any(
        invalid in e.lower() for invalid in ['example.com', 'test', 'error', 'yourdomain.com', 'email@', 'sentry.io'])]

    if valid_emails:
        email = valid_emails[0] # Take the first valid one

    # Extract pain points
    pain_pattern = r'(?:Pain Points?|Challenges?|Issues?|Analysis|Opportunities)(?:\s*:|\s*\n)(.*?)(?:Email:|Contact:|Conclusion:|\Z)'
    pain_matches = re.findall(pain_pattern, result, re.IGNORECASE | re.DOTALL)

    if pain_matches:
        pain_points = pain_matches[0].strip()
        pain_points = re.sub(r'^[\s*-]+', '', pain_points, flags=re.MULTILINE).strip()
    else:
        if email and email in result:
            pain_points = result.split(email, 1)[1].strip()
            pain_points = re.sub(r'^(?:Pain Points?|Challenges?|Issues?|Analysis|Opportunities)(?:\s*:|\s*\n)', '', pain_points, flags=re.IGNORECASE).strip()
            pain_points = re.sub(r'^[\s*-]+', '', pain_points, flags=re.MULTILINE).strip()
        else:
            pain_points = result.strip() # Use everything

        # Attempt to remove the email itself if it's the only thing left
        if email and pain_points == email:
             pain_points = ""

    if not pain_points.strip():
        # If still empty, check if the *original* result contained *any* text besides the email
        text_without_email = result.replace(email, "").strip() if email else result.strip()
        if text_without_email and len(text_without_email) > 10: # Avoid setting noise as pain points
             pain_points = text_without_email
        else:
             pain_points = "No specific pain points identified in the output."


    return {"email": email, "pain_points": pain_points}


def extract_companies_from_url(url, agents, extraction_task):
    """
    Extract company information from a given URL.

    Args:
        url: URL to process
        agents: Dictionary of initialized agents
        extraction_task: Task object for extraction

    Returns:
        List of extracted company data
    """
    extracted_company_data = []
    try:
        # Set up extraction crew
        extraction_crew = Crew(
            agents=[agents['research']],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=False
        )

        # Execute extraction
        logger.debug(f"  Kicking off extraction crew for {url}...")
        extraction_result_object = extraction_crew.kickoff()
        logger.debug(f"  Extraction crew finished for {url}.")

        if extraction_result_object:
            raw_output = None
            if isinstance(extraction_result_object, CrewOutput):
                raw_output = extraction_result_object.raw
                logger.debug(f"  Received CrewOutput object from extraction. Extracted raw output.")
            elif isinstance(extraction_result_object, str):
                raw_output = extraction_result_object
                logger.debug(f"  Received string output directly from extraction.")
            else:
                logger.warning(f"  Extraction crew returned unexpected type: {type(extraction_result_object)} for {url}")

            if raw_output:
                extracted_company_data = parse_company_website_list(raw_output)
                # Use INFO level for successful extraction count for better visibility
                if extracted_company_data:
                    logger.info(f"  Extracted {len(extracted_company_data)} potential HR company/website pairs from {url}.")
                else:
                    logger.info(f"  Parsing company list returned no results for {url}.")
            else:
                 logger.warning(f"  Extraction crew returned output, but raw string could not be extracted for {url}.")
        else:
            logger.warning(f"  Extraction crew returned no output object for {url}.")
    except Exception as e:
        logger.error(f"  Error during company extraction process for '{url}': {e}", exc_info=True)

    return extracted_company_data


# --- analyze_company Function Definition ---
# Remove analysis_task from parameters
# MODIFIED: Added 'category' parameter
def analyze_company(company_name, company_website, agents, category: str):
    """
    Analyze a company to find email and pain points, including a review cycle,
    using agents specific to the company's category (HR or NE_B2B).

    Args:
        company_name: Name of the company
        company_website: Website URL of the company
        agents: Dictionary of ALL initialized agents (research, hr_analysis, etc.)
        category: The determined category of the company ('HR' or 'NE_B2B')

    Returns:
        Dictionary with company analysis data including the category.
    """
    initial_analysis_results = {
        "name": company_name,
        "website": company_website,
        "pain_points": "Initial analysis did not run",
        "contact_email": "",
        "category": category # Store category in results from the start
    }
    final_company_data = initial_analysis_results.copy()

    # --- Validate Category ---
    # Determine agent keys based on category
    if category == 'HR':
        analysis_agent_key = 'hr_analysis'
        reviewer_agent_key = 'hr_reviewer'
    elif category == 'NE_B2B':
        analysis_agent_key = 'ne_b2b_analysis'
        reviewer_agent_key = 'ne_b2b_reviewer'
    else:
        logger.error(f"  Invalid category '{category}' provided for company '{company_name}'. Skipping analysis.")
        final_company_data["pain_points"] = f"Analysis skipped - Invalid category: {category}"
        return final_company_data

    # --- Get Specific Agents ---
    analysis_agent = agents.get(analysis_agent_key)
    reviewer_agent = agents.get(reviewer_agent_key)

    if not analysis_agent or not isinstance(analysis_agent, Agent):
        logger.error(f"  {analysis_agent_key} agent not found or invalid for company '{company_name}'. Skipping analysis.")
        final_company_data["pain_points"] = f"Analysis skipped - {analysis_agent_key} agent missing"
        return final_company_data

    if not reviewer_agent or not isinstance(reviewer_agent, Agent):
        # Log warning but proceed with analysis if reviewer is missing, skip review later
        logger.warning(f"  {reviewer_agent_key} agent not found or invalid for company '{company_name}'. Review cycle will be skipped.")
        # Don't return yet, allow analysis to run


    # --- Main Try/Except Block for the whole analysis process ---
    try:
        # === Stage 1: Initial Analysis ===
        logger.info(f"      >>> Starting {category} analysis for '{company_name}' using {analysis_agent_key}...") # Indicate category

        # --- CHANGE: Create analysis_task using the SELECTED agent ---
        logger.debug(f"      Creating analysis task internally for {company_name} using {analysis_agent_key}")
        # Pass the specific agent instance determined by the category
        analysis_task = create_analysis_task(company_name, company_website, analysis_agent)

        # Check if task creation failed
        if analysis_task is None:
             logger.error(f"      Failed to create analysis task for '{company_name}' using {analysis_agent_key}. Skipping analysis.")
             final_company_data["pain_points"] = "Analysis failed - Task creation error"
             return final_company_data


        # --- (Optional DEBUG block - can be removed if stable) ---
        # try:
        #     logger.debug(f"      DEBUG: --- Checking objects before creating Analysis Crew for '{company_name}' ({category}) ---")
        #     logger.debug(f"      DEBUG: Using analysis_agent: {analysis_agent.role}")
        #     logger.debug(f"      DEBUG: analysis_task object type: {type(analysis_task)}")
        #     if isinstance(analysis_task, Task):
        #          assigned_agent = getattr(analysis_task, 'agent', None)
        #          logger.debug(f"      DEBUG: analysis_task.agent role: {getattr(assigned_agent, 'role', 'Not Found')}")
        #     logger.debug(f"      DEBUG: --- End checks ---")
        # except Exception as debug_e:
        #     logger.error(f"      DEBUG: Error during detailed debug logging: {debug_e}")
        # --- (End Optional DEBUG block) ---


        # Create the analysis crew using the SELECTED agent
        analysis_crew = Crew(
            agents=[analysis_agent], # Use the specific agent
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False
        )
        analysis_result_object = analysis_crew.kickoff()
        logger.debug(f"      <<< Initial {category} analysis finished for '{company_name}'.")

        # Parse initial results (Handling CrewOutput - no changes here)
        initial_email = ""
        initial_pain_points = "Initial analysis failed: No output object."
        # ... (parsing logic remains the same) ...
        if analysis_result_object:
            raw_output = None
            if isinstance(analysis_result_object, CrewOutput):
                raw_output = analysis_result_object.raw
            elif isinstance(analysis_result_object, str):
                raw_output = analysis_result_object

            if raw_output:
                parsed_initial = parse_analysis_results(raw_output)
                initial_email = parsed_initial.get('email', '')
                initial_pain_points = parsed_initial.get('pain_points', 'Initial analysis parsing failed')
                logger.debug(f"      Parsed initial {category} analysis - Email: '{initial_email}', Points: '{initial_pain_points[:100]}...'")
            else:
                logger.warning(f"      Initial {category} analysis returned output, but raw string could not be extracted for {company_name}.")
                initial_pain_points = "Initial analysis failed: Could not extract raw output."
        else:
             logger.warning(f"      Initial {category} analysis crew returned no output object for {company_name}.")


        # Update dict with initial findings
        final_company_data["contact_email"] = initial_email
        final_company_data["pain_points"] = initial_pain_points

        # === Stage 2: Review Cycle ===
        # Check if reviewer agent is valid before proceeding
        if not reviewer_agent or not isinstance(reviewer_agent, Agent):
             logger.warning(f"      Skipping review cycle for '{company_name}' because {reviewer_agent_key} agent is missing or invalid.")
        # Check initial_pain_points more carefully before skipping
        elif not initial_pain_points or initial_pain_points.startswith("Initial analysis failed") or initial_pain_points.startswith("Analysis failed"):
            logger.warning(f"      Skipping review cycle for '{company_name}' due to initial analysis failure or lack of points.")
        else:
            logger.info(f"      >>> Starting {category} pain point review cycle for '{company_name}' using {reviewer_agent_key}...")

            # --- CHANGE: Create review_task using the SELECTED reviewer agent ---
            review_task = create_review_task(
                company_name,
                company_website,
                initial_pain_points,
                reviewer_agent # Pass the specific reviewer agent
            )

            # Check if task creation failed
            if review_task is None:
                 logger.error(f"      Failed to create review task for '{company_name}' using {reviewer_agent_key}. Skipping review.")
            else:
                # Execute the task directly using the SELECTED reviewer agent
                # Inner try/except specifically for the review task execution
                try:
                    review_result_output = reviewer_agent.execute_task(review_task)
                    logger.debug(f"      <<< {category} Review cycle finished for '{company_name}'.")

                    # Parse the review results (Handling CrewOutput - no changes here)
                    # ... (parsing logic remains the same) ...
                    if review_result_output:
                        review_raw_output = None
                        if isinstance(review_result_output, CrewOutput):
                            review_raw_output = review_result_output.raw
                        elif isinstance(review_result_output, str):
                             review_raw_output = review_result_output

                        if review_raw_output:
                            # Use the existing parser, assuming output format is consistent
                            parsed_review = parse_analysis_results(review_raw_output)
                            reviewed_pain_points = parsed_review.get('pain_points')

                            # Only update if review provided valid, different points
                            if reviewed_pain_points and not reviewed_pain_points.startswith("Analysis failed") and reviewed_pain_points != initial_pain_points:
                                logger.info(f"      {category} Review cycle provided refined pain points for '{company_name}'.")
                                final_company_data["pain_points"] = reviewed_pain_points # Update with refined points
                            elif reviewed_pain_points == initial_pain_points:
                                logger.info(f"      {category} Review cycle validated initial pain points for '{company_name}'.")
                            else:
                                logger.warning(f"      {category} Review cycle output parsing failed or yielded no points for {company_name}. Using initial points.")
                        else:
                             logger.warning(f"      {category} Review task returned output, but raw string could not be extracted for {company_name}. Using initial points.")
                    else:
                        logger.warning(f"      {category} Review task returned no output object for {company_name}. Using initial points.")

                except Exception as review_err:
                    logger.error(f"      Error during {category} review task execution for '{company_name}': {review_err}", exc_info=True)
                    logger.warning("      Using initial pain points due to review execution error.")

        # Return the potentially updated data (includes category)
        return final_company_data

    # --- Corresponds to the main try block at the beginning of the function ---
    except Exception as e:
        logger.error(f"      Error during overall {category} company analysis process for '{company_name}': {e}", exc_info=True)
        # Return dictionary with error message, using initial values if available
        error_data = final_company_data.copy() # Already includes category
        error_data["pain_points"] = f"Analysis failed ({category}): Exception - {type(e).__name__}: {str(e)}" # Include category in error
        if not error_data.get("contact_email"):
             error_data["contact_email"] = ""
        return error_data

    # Add return here for completeness outside the try/except
    return final_company_data
