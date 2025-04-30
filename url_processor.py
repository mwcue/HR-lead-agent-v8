# url_processor.py
import logging
import ast
import re
# Import CrewOutput to check its type
from crewai import Crew, Process, CrewOutput, Agent
from config import Config
# Assuming the parser is now in utils/parser.py as per standard structure
# If it's elsewhere (like company_extractor.py), change the import path.
from utils.parser import parse_url_list

# Configure logging
logger = logging.getLogger(__name__)

# Note: The actual definition of parse_url_list is assumed to be imported above.
# If it needs to be defined here, it would go here.

def perform_search(agents, tasks):
    """
    Execute search tasks to find relevant URLs using the Research Agent.

    Args:
        agents: Dictionary of initialized agents (expecting 'research')
        tasks: List of search-related tasks [plan_task, execute_task]

    Returns:
        List of URLs found during search
    """
    url_list = []
    try:
        logger.info("--- Kicking off Search Tasks (Plan & Execute) ---")

        # Basic check on tasks list
        if not isinstance(tasks, list) or len(tasks) < 1:
            logger.error("Invalid or empty tasks list provided to perform_search.")
            return []
        if 'research' not in agents or not isinstance(agents.get('research'), Agent):
             logger.error("Research agent not found or invalid in 'agents' dict.")
             return []

        search_crew = Crew(
            agents=[agents['research']], # Uses the Research Agent
            tasks=tasks, # Pass both plan and execute tasks
            process=Process.sequential,
            verbose=False # Set to True temporarily IF you want detailed CrewAI step logs
        )

        # Execute search (both planning and execution if two tasks provided)
        search_results_object = search_crew.kickoff()
        logger.info("--- Search Tasks Finished ---")

        raw_output = None # Initialize raw_output
        if search_results_object:
            if isinstance(search_results_object, CrewOutput):
                raw_output = search_results_object.raw
                logger.debug("Extracted raw output from CrewOutput object.")
            elif isinstance(search_results_object, str):
                raw_output = search_results_object
                logger.debug("Received string output directly.")
            else:
                logger.warning(f"Search Crew returned unexpected type: {type(search_results_object)}")

            # --- >>> ADDED THIS CRITICAL LOGGING <<< ---
            # Check if raw_output was successfully extracted before logging/parsing
            if raw_output is not None: # Check specifically for None
                # Use ERROR level to make sure it's visible even with INFO default
                logger.error(f"!!! RAW OUTPUT from Research Agent BEFORE parsing URL list:\n---\n{raw_output}\n---")
                # Now, attempt parsing using the imported function
                url_list = parse_url_list(raw_output) # Make sure parse_url_list function exists
                logger.debug(f"Full URL list found by search execution (post-parsing attempt): {url_list}")
                # Log change to reflect potential parsing failure
                logger.info(f"Attempted to parse URLs. Extracted {len(url_list)} URLs: {url_list[:5]}..." if url_list else 'None (Parsing Failed or No URLs in Output)')
            else:
                logger.error("!!! NO RAW OUTPUT string could be extracted from search_results_object.")
                logger.warning("Cannot parse URLs because raw output string could not be extracted.")
            # --- >>> END CRITICAL LOGGING <<< ---

        else: # This else corresponds to 'if search_results_object:'
            logger.warning("Search Crew did not return any output object.")

    except Exception as e:
        logger.error(f"\nSearch Crew execution or parsing error: {e}", exc_info=True) # Added exc_info

    return url_list
