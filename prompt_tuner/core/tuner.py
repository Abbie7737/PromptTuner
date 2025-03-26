"""
Core prompt tuning implementation.
"""
import asyncio
import os
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

from pydantic_ai import Agent
from prompt_tuner.utils.lmstudio import LMStudioManager
from prompt_tuner.utils.prompt_loader import PromptLoader


class PromptTuner:
    """
    A system that uses a large LLM to optimize prompts for small LLMs.
    """
    
    def __init__(self, env_file: str = ".env") -> None:
        """
        Initialize the prompt tuner.
        
        Args:
            env_file: Path to the .env file with configuration
        """
        self.lmstudio = LMStudioManager(env_file)
        self.prompt_loader = PromptLoader()
        self.num_initial_prompts = int(os.environ.get("NUM_INITIAL_PROMPTS", "5"))
        self.results_dir = os.environ.get("RESULTS_DIR", "results")
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def tune(
        self, 
        original_prompt: str, 
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tune a prompt for better performance on a small LLM.
        
        Args:
            original_prompt: The original prompt to optimize
            task_description: Optional description of what the prompt should achieve
            
        Returns:
            Dictionary with optimization results
        """
        # Start tracking results for final report
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "run_id": run_id,
            "original_prompt": original_prompt,
            "task_description": task_description,
            "initial_prompts": [],
            "evaluation_round_1": [],
            "refined_prompts": [],
            "evaluation_round_2": [],
            "best_prompt": None,
            "explanation": None
        }
        
        # Phase 1: Generate initial prompt variations
        print(f"Generating {self.num_initial_prompts} initial prompt variations...")
        print(f"Using large model: {self.lmstudio.large_model.model}")
        print(f"This may take a few minutes for reasoning models...")
        
        initial_prompts = await self._generate_prompt_variations(
            original_prompt, 
            task_description,
            self.num_initial_prompts
        )
        print(f"Generated {len(initial_prompts)} prompt variations")
        results["initial_prompts"] = initial_prompts
        
        # Phase 2: Evaluate initial prompts
        print("Evaluating initial prompt variations...")
        print(f"Running each prompt on the small model: {self.lmstudio.small_model.model}")
        print(f"Then evaluating responses with the large model...")
        evaluations = await self._evaluate_prompts(initial_prompts, original_prompt, task_description)
        print(f"Completed evaluation of {len(evaluations)} prompts")
        results["evaluation_round_1"] = evaluations
        
        # Sort by score and take top 2
        sorted_evals = sorted(evaluations, key=lambda x: x["score"], reverse=True)
        
        # Print scores to help with debugging
        print("\nScores from first evaluation round:")
        for i, eval_item in enumerate(sorted_evals):
            print(f"  Prompt {i+1}: {eval_item['score']:.1f}/10")
            
        top_prompts = [eval_item["prompt"] for eval_item in sorted_evals[:2]]
        
        # Phase 3: Refine best prompts
        print("Refining top 2 prompts...")
        print(f"Using large model to improve the best prompts based on evaluation...")
        refined_prompts = []
        for i, prompt in enumerate(top_prompts):
            print(f"Refining prompt {i+1}/2...")
            refined = await self._refine_prompt(
                prompt, 
                sorted_evals[i]["response"], 
                sorted_evals[i]["explanation"],
                original_prompt,
                task_description
            )
            refined_prompts.append(refined)
            print(f"Refinement {i+1} complete")
        print(f"Refined {len(refined_prompts)} prompts")
        results["refined_prompts"] = refined_prompts
        
        # Phase 4: Evaluate refined prompts
        print("Evaluating refined prompts...")
        print(f"Testing refined prompts on the small model and evaluating results...")
        refined_evals = await self._evaluate_prompts(refined_prompts, original_prompt, task_description)
        print(f"Final evaluation complete")
        results["evaluation_round_2"] = refined_evals
        
        # Phase 5: Select and explain best prompt
        print("Selecting and explaining the best prompt...")
        
        # Get best scores from each round
        sorted_initial = sorted(evaluations, key=lambda x: x["score"], reverse=True)
        sorted_refined = sorted(refined_evals, key=lambda x: x["score"], reverse=True)
        
        # Print scores from both rounds for comparison
        print("\nScores from first evaluation round:")
        for i, eval_item in enumerate(sorted_initial):
            print(f"  Initial prompt {i+1}: {eval_item['score']:.1f}/10")
            
        print("\nScores from final evaluation round:")
        for i, eval_item in enumerate(sorted_refined):
            print(f"  Refined prompt {i+1}: {eval_item['score']:.1f}/10")
        
        # Compare best scores from each round to determine the overall winner
        best_initial_score = sorted_initial[0]["score"] if sorted_initial else 0
        best_refined_score = sorted_refined[0]["score"] if sorted_refined else 0
        
        print(f"\nBest initial prompt score: {best_initial_score:.1f}/10")
        print(f"Best refined prompt score: {best_refined_score:.1f}/10")
        
        # Select the best prompt from either round
        if best_initial_score >= best_refined_score:
            best_prompt = sorted_initial[0]["prompt"]
            best_score = best_initial_score
            best_response = sorted_initial[0]["response"]
            print(f"Initial prompt selected as best with score: {best_score:.1f}/10")
        else:
            best_prompt = sorted_refined[0]["prompt"]
            best_score = best_refined_score
            best_response = sorted_refined[0]["response"]
            print(f"Refined prompt selected as best with score: {best_score:.1f}/10")
        results["best_prompt"] = best_prompt
        
        # Generate explanation for why the best prompt works well
        print("Generating detailed explanation of why this prompt works well...")
        explanation = await self._explain_best_prompt(
            best_prompt,
            original_prompt,
            task_description,
            best_response  # Use the response corresponding to the best prompt
        )
        print("Explanation generated")
        results["explanation"] = explanation
        
        # Save raw results as JSON for later analysis first (in case report generation fails)
        json_path = os.path.join(self.results_dir, f"prompt_tuning_{run_id}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Raw data saved to: {json_path}")
            
        # Generate final report (with error handling)
        print("Generating final report...")
        try:
            report = await self._generate_report(results)
            report_path = os.path.join(self.results_dir, f"prompt_tuning_{run_id}.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"Error during report generation and saving: {str(e)}")
            print("A simplified report will be generated instead")
            
            # Create a more detailed report without using the API
            simple_report_path = os.path.join(self.results_dir, f"prompt_tuning_{run_id}_simple.md")
            
            # Extract evaluation scores to include in the report
            initial_eval_summary = []
            for i, eval_item in enumerate(results["evaluation_round_1"]):
                initial_eval_summary.append(f"{i+1}. Score: {eval_item['score']:.1f}/10 - `{eval_item['prompt'][:100]}...`")
            
            refined_eval_summary = []
            for i, eval_item in enumerate(results["evaluation_round_2"]):
                refined_eval_summary.append(f"{i+1}. Score: {eval_item['score']:.1f}/10 - `{eval_item['prompt'][:100]}...`")
            
            # Find the best prompts from both rounds
            best_initial_score = 0
            best_refined_score = 0
            
            if results["evaluation_round_1"]:
                sorted_initial = sorted(results["evaluation_round_1"], key=lambda x: x["score"], reverse=True)
                best_initial_score = sorted_initial[0]["score"]
                
            if results["evaluation_round_2"]:
                sorted_refined = sorted(results["evaluation_round_2"], key=lambda x: x["score"], reverse=True)
                best_refined_score = sorted_refined[0]["score"]
                
            # Determine which round had the best prompt
            best_score = max(best_initial_score, best_refined_score)
            
            # Add a note about which round produced the best prompt
            if best_initial_score >= best_refined_score:
                best_round = "first evaluation round"
            else:
                best_round = "refinement round"
            
            simple_report = f"""# Prompt Tuning Report - {run_id}

## Original Prompt
```
{results["original_prompt"]}
```

## Evaluation Process

### First Evaluation Round
{chr(10).join(initial_eval_summary)}

### Final Evaluation
{chr(10).join(refined_eval_summary)}

## Best Prompt (Score: {best_score:.1f}/10)
This prompt was selected from the {best_round}.

```
{results["best_prompt"]}
```

## Comparison of Best Scores
- Best score from initial prompts: {best_initial_score:.1f}/10
- Best score from refined prompts: {best_refined_score:.1f}/10

## Explanation
{results["explanation"]}

*Note: This is a simplified report generated due to an error during full report generation.*
"""
            with open(simple_report_path, "w") as f:
                f.write(simple_report)
            print(f"Simplified report saved to: {simple_report_path}")
            report_path = simple_report_path
        
        return {
            "best_prompt": best_prompt,
            "explanation": explanation,
            "report_path": report_path,
            "json_path": json_path
        }
    
    async def _generate_prompt_variations(
        self, 
        original_prompt: str, 
        task_description: Optional[str],
        num_variations: int
    ) -> List[str]:
        """
        Generate variations of the original prompt.
        
        Args:
            original_prompt: The original prompt
            task_description: Optional description of the task
            num_variations: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        generator_prompt = self.prompt_loader.format(
            "generator",
            original_prompt=original_prompt,
            task_description=task_description or "No specific task description provided",
            num_variations=num_variations
        )
        
        messages = [
            {"role": "system", "content": generator_prompt},
            {"role": "user", "content": f"Generate {num_variations} variations of this prompt for a small language model: {original_prompt}"}
        ]
        
        response = await self.lmstudio.run_on_large_model(
            messages=messages,
            temperature=0.7
        )
        
        # Extract variations from response
        content = response["choices"][0]["message"]["content"]
        
        # Parse variations (expecting numbered list)
        variations = []
        current_variation = ""
        lines = content.split("\n")
        
        for line in lines:
            # Check for numbered line starts (1., 2., etc.)
            if line.strip() and line[0].isdigit() and line[1:].startswith(". "):
                if current_variation:
                    variations.append(current_variation.strip())
                current_variation = line[line.find(" ")+1:]
            else:
                current_variation += "\n" + line
        
        # Add the last variation
        if current_variation:
            variations.append(current_variation.strip())
        
        # Ensure we have the right number of variations
        variations = variations[:num_variations]
        
        return variations
    
    async def _evaluate_prompts(
        self, 
        prompts: List[str], 
        original_prompt: str,
        task_description: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate prompts by running them on the small model and scoring responses.
        
        Args:
            prompts: List of prompts to evaluate
            original_prompt: The original prompt for reference
            task_description: Optional description of the task
            
        Returns:
            List of evaluation results with scores and explanations
        """
        evaluations = []
        
        for i, prompt in enumerate(prompts):
            print(f"Evaluating prompt {i+1}/{len(prompts)}...")
            
            # Run the prompt on the small model
            messages = [
                {"role": "system", "content": prompt}
            ]
            
            # If no task description, use the original prompt as the user message
            user_message = task_description or original_prompt
            messages.append({"role": "user", "content": user_message})
            
            small_model_response = await self.lmstudio.run_on_small_model(
                messages=messages,
                temperature=0.2
            )
            response_content = small_model_response["choices"][0]["message"]["content"]
            
            # Have the large model evaluate the response
            evaluator_prompt = self.prompt_loader.format(
                "evaluator",
                original_prompt=original_prompt,
                task_description=task_description or "No specific task description provided",
                prompt_variation=prompt,
                small_model_response=response_content
            )
            
            eval_messages = [
                {"role": "system", "content": evaluator_prompt},
                {"role": "user", "content": "Evaluate the small model's response."}
            ]
            
            evaluation = await self.lmstudio.run_on_large_model(
                messages=eval_messages,
                temperature=0.2
            )
            
            eval_content = evaluation["choices"][0]["message"]["content"]
            print(f"Received evaluation from large model")
            
            # Parse score and explanation from evaluation
            score = 0
            explanation = eval_content
            
            # Try to extract score (expecting format like "Score: 8/10")
            for line in eval_content.split("\n"):
                if line.lower().startswith("score:"):
                    try:
                        # Extract number before the slash if present
                        score_text = line.split(":")[1].strip()
                        if "/" in score_text:
                            score = float(score_text.split("/")[0].strip())
                        else:
                            score = float(score_text)
                        print(f"Extracted score: {score}")
                        break
                    except (ValueError, IndexError):
                        # If we can't parse the score, use 0
                        print(f"Error parsing score from: {line}")
                        score = 0
            
            if score == 0:
                print("WARNING: Could not extract a score > 0 from evaluation")
                print(f"First few lines of evaluation: {eval_content[:200]}...")
                
                # Try one more parsing approach - look for any number followed by /10
                import re
                score_matches = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*10', eval_content)
                if score_matches:
                    score = float(score_matches[0])
                    print(f"Extracted score using regex: {score}")
            
            evaluations.append({
                "prompt": prompt,
                "response": response_content,
                "evaluation": eval_content,
                "score": score,
                "explanation": explanation
            })
            
            print(f"Completed evaluation {i+1} with score: {score}")
        
        return evaluations
    
    async def _refine_prompt(
        self,
        prompt: str,
        response: str,
        evaluation: str,
        original_prompt: str,
        task_description: Optional[str]
    ) -> str:
        """
        Refine a prompt based on its evaluation.
        
        Args:
            prompt: The prompt to refine
            response: The small model's response to the prompt
            evaluation: The evaluation of the response
            original_prompt: The original prompt for reference
            task_description: Optional description of the task
            
        Returns:
            The refined prompt
        """
        refiner_prompt = self.prompt_loader.format(
            "refiner",
            original_prompt=original_prompt,
            task_description=task_description or "No specific task description provided",
            prompt_to_refine=prompt,
            small_model_response=response,
            evaluation=evaluation
        )
        
        messages = [
            {"role": "system", "content": refiner_prompt},
            {"role": "user", "content": "Refine this prompt to improve performance on the small model."}
        ]
        
        response = await self.lmstudio.run_on_large_model(
            messages=messages,
            temperature=0.5
        )
        
        content = response["choices"][0]["message"]["content"]
        
        # Extract the refined prompt
        refined_prompt = content
        
        # If the response has clear delimiters, extract between them
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:  # At least one code block
                refined_prompt = parts[1]
                if refined_prompt.startswith("prompt") or refined_prompt.startswith("system"):
                    refined_prompt = refined_prompt.split("\n", 1)[1]
        
        return refined_prompt.strip()
    
    async def _explain_best_prompt(
        self,
        best_prompt: str,
        original_prompt: str,
        task_description: Optional[str],
        best_response: str
    ) -> str:
        """
        Generate an explanation for why the best prompt works well.
        
        Args:
            best_prompt: The best performing prompt
            original_prompt: The original prompt for reference
            task_description: Optional description of the task
            best_response: The small model's response to the best prompt
            
        Returns:
            Explanation of why the best prompt works well
        """
        explainer_prompt = self.prompt_loader.format(
            "explainer",
            best_prompt=best_prompt,
            original_prompt=original_prompt,
            task_description=task_description or "No specific task description provided",
            best_response=best_response
        )
        
        messages = [
            {"role": "system", "content": explainer_prompt},
            {"role": "user", "content": "Explain why this is the best prompt."}
        ]
        
        response = await self.lmstudio.run_on_large_model(
            messages=messages,
            temperature=0.3
        )
        
        explanation = response["choices"][0]["message"]["content"]
        return explanation
    
    async def _generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a Markdown report of the prompt tuning process.
        
        Args:
            results: Dictionary with all results from the tuning process
            
        Returns:
            Markdown formatted report
        """
        try:
            # Create a simplified version of the data to avoid token limits
            simplified_eval_round_1 = []
            for eval_item in results["evaluation_round_1"]:
                simplified_eval_round_1.append({
                    "prompt": eval_item["prompt"],
                    "score": eval_item["score"],
                })
                
            simplified_eval_round_2 = []
            for eval_item in results["evaluation_round_2"]:
                simplified_eval_round_2.append({
                    "prompt": eval_item["prompt"],
                    "score": eval_item["score"],
                })
                
            # Determine which round had the best prompt
            best_initial_score = 0
            best_refined_score = 0
            best_round = "unknown"
            
            if simplified_eval_round_1:
                sorted_initial = sorted(simplified_eval_round_1, key=lambda x: x["score"], reverse=True)
                best_initial_score = sorted_initial[0]["score"]
                
            if simplified_eval_round_2:
                sorted_refined = sorted(simplified_eval_round_2, key=lambda x: x["score"], reverse=True)
                best_refined_score = sorted_refined[0]["score"]
            
            if best_initial_score >= best_refined_score:
                best_round = "first evaluation round"
            else:
                best_round = "refinement round"
                
            report_prompt = self.prompt_loader.format(
                "report",
                run_id=results["run_id"],
                original_prompt=results["original_prompt"],
                task_description=results["task_description"] or "No specific task description provided",
                num_initial_prompts=len(results["initial_prompts"]),
                best_prompt=results["best_prompt"],
                explanation=results["explanation"],
                initial_prompts=json.dumps(results["initial_prompts"]),
                evaluation_round_1=json.dumps(simplified_eval_round_1),
                refined_prompts=json.dumps(results["refined_prompts"]),
                evaluation_round_2=json.dumps(simplified_eval_round_2),
                best_initial_score=best_initial_score,
                best_refined_score=best_refined_score,
                best_round=best_round
            )
            
            messages = [
                {"role": "system", "content": report_prompt},
                {"role": "user", "content": "Generate a detailed report of the prompt tuning process."}
            ]
            
            response = await self.lmstudio.run_on_large_model(
                messages=messages,
                temperature=0.3,
                max_tokens=4096
            )
            
            report = response["choices"][0]["message"]["content"]
            return report
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            # Create a simple fallback report
            fallback_report = f"""# Prompt Tuning Report - {results["run_id"]}

## Original Prompt
```
{results["original_prompt"]}
```

## Best Prompt
```
{results["best_prompt"]}
```

## Explanation
{results["explanation"]}

*Note: This is a simplified report. Full report generation failed with error: {str(e)}*
"""
            return fallback_report