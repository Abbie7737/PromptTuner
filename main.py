#!/usr/bin/env python
"""
Prompt Tuner - A tool for optimizing prompts for small language models.

This script is the entry point for the prompt tuning system.
"""
import asyncio
import argparse
import os
import sys
from typing import Optional

from prompt_tuner.core.tuner import PromptTuner


async def run(
    prompt: str, 
    task_description: Optional[str] = None,
    env_file: str = ".env"
) -> None:
    """
    Run the prompt tuner on a given prompt.
    
    Args:
        prompt: The prompt to optimize
        task_description: Optional description of what the prompt should achieve
        env_file: Path to the environment file
    """
    tuner = PromptTuner(env_file)
    results = await tuner.tune(prompt, task_description)
    
    print("\n" + "="*50)
    print("✨ Prompt tuning complete! ✨")
    print(f"Best prompt saved to: {results['report_path']}")
    print(f"Full results saved to: {results['json_path']}")
    print("="*50)
    print("\nBest Prompt:")
    print("-"*50)
    print(results["best_prompt"])
    print("-"*50)
    print("\nExplanation:")
    print("-"*50)
    print(results["explanation"])
    print("-"*50)


def main() -> int:
    """
    Parse command line arguments and run the prompt tuner.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Tune prompts for better performance on small language models."
    )
    parser.add_argument(
        "prompt", 
        nargs="?",
        help="The prompt to optimize (or provide with -f/--file)"
    )
    parser.add_argument(
        "-f", "--file",
        help="Read prompt from a file instead of command line"
    )
    parser.add_argument(
        "-t", "--task",
        help="Description of what the prompt should achieve"
    )
    parser.add_argument(
        "-e", "--env",
        default=".env",
        help="Path to environment file (default: .env)"
    )
    
    args = parser.parse_args()
    
    # Get prompt from file or command line
    prompt = args.prompt
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: Prompt file not found: {args.file}")
            return 1
        
        with open(args.file, "r") as f:
            prompt = f.read()
    
    if not prompt:
        parser.print_help()
        return 1
    
    asyncio.run(run(prompt, args.task, args.env))
    return 0


if __name__ == "__main__":
    sys.exit(main())