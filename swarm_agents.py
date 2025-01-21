import heapq
import random
import logging
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import asyncio
from logging.handlers import RotatingFileHandler
from swarm_utils import *

logging.basicConfig(
    filename=f'./swarm_logs/adaptive_hpc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


health_logger = logging.getLogger('health_monitor')
health_logger.setLevel(logging.INFO)

# Create a file handler with a unique filename
health_handler = logging.FileHandler(f'./swarm_logs/health_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
health_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
health_handler.setFormatter(health_formatter)

health_logger.addHandler(health_handler)
health_logger.propagate = False


@dataclass
class SystemState:
    simulation_time: float
    scheduler_state: Dict[str, Any]
    resource_agents_state: List[Dict[str, Any]]
    pending_jobs: List[Dict[str, Any]]
    assigned_jobs: List[Dict[str, Any]]
    completed_jobs: List[Dict[str, Any]]

class SystemHealthCheckAgent:
    """Monitors the health and state of the system in real-time."""

    def __init__(self, scheduler: 'AdvancedSchedulerAgent', interval:float=5.0, logger: Optional[logging.Logger]=None):
        """
        Initialize the health check agent with the scheduler and monitoring interval.
        Params:
            scheduler: Reference to the scheduler agent.
            interval: Time interval (in seconds) between health checks.
            logger: Logger instance for health monitoring.
        """
        self.scheduler = scheduler
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.logger = logger if logger else logging.getLogger('health_monitor')
    
    
    async def start(self):
        """Start the health check monitoring agent."""
        self.running = True
        self.task = asyncio.create_task(self.monitor())
        self.logger.info("SystemHealthCheckAgent started.")

    async def stop(self):
        """Stop the health check monitoring agent."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.logger.info("SystemHealthCheckAgent stopped.")

    async def monitor(self):
        """Periodically monitor and log system health."""
        while self.running:
            try:  
                system_state = self.collect_system_state()  
                self.log_system_state(system_state)  
            except Exception as e:  
                self.logger.error(f"Error during monitoring: {e}")  
            await asyncio.sleep(self.interval)

    def collect_system_state(self) -> SystemState:
        """Gather current state of the system."""
        scheduler_state = {
            "current_time": self.scheduler.current_time,
            "event_queue_length": len(self.scheduler.event_queue),
            # "event_queue":self.scheduler.event_queue,
        }

        resource_agents_state = []
        for agent in self.scheduler.agents:
            agent_state = {
                "agent_id": agent.agent_id,
                "total_nodes": agent.num_nodes,
                "available_nodes": agent.available_nodes,
                "current_load": agent.calculate_load(),
                "assigned_jobs_count": len(agent.assigned_jobs),
                "completed_jobs_count": len(agent.completed_jobs)
            }
            resource_agents_state.append(agent_state)

        pending_jobs = [{
            "job_id": job.job_id,
            "description": job.description,
            "min_nodes": job.execution_profile.min_nodes,
            "max_nodes": job.execution_profile.max_nodes,
            "predicted_runtime": job.predicted_runtime            
        } for job in self.scheduler.pending_jobs]

        assigned_jobs = [{  
            "job_id": job.job_id,  
            "assigned_agent": next((agent.agent_id for agent in self.scheduler.agents if job in agent.assigned_jobs), None),  
            "allocated_nodes": job.assigned_nodes,  
            "predicted_runtime": job.predicted_runtime,  
            "start_time": job.start_time  
        } for agent in self.scheduler.agents for job in agent.assigned_jobs]  

        completed_jobs = [{  
            "job_id": job.job_id,  
            "assigned_agent": next((agent.agent_id for agent in self.scheduler.agents if job in agent.completed_jobs), None),  
            "actual_runtime": job.actual_runtime,  
            "completion_time": job.completion_time  
        } for job in self.scheduler.completed_jobs]

        return SystemState(  
            simulation_time=self.scheduler.current_time,  
            scheduler_state=scheduler_state,  
            resource_agents_state=resource_agents_state,  
            pending_jobs=pending_jobs,  
            assigned_jobs=assigned_jobs,  
            completed_jobs=completed_jobs  
        )
    
    def log_system_state(self, state: SystemState):
        """Log the collected system state."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "simulation_time": state.simulation_time,
            "scheduler_state": state.scheduler_state,
            "resource_agents_state": state.resource_agents_state,
            "pending_jobs": state.pending_jobs,
            "assigned_jobs": state.assigned_jobs,
            "completed_jobs": state.completed_jobs
        }
        self.logger.info(f"System health: {json.dumps(log_entry, indent=2)}")

@dataclass
class Event:
    """Enhanced representation of discrete simulation events with comprehensive temporal tracking."""
    time: float
    action: str
    agent_id: Optional[int] = None
    job: Optional['AdaptiveJob'] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __lt__(self, other: 'Event') -> bool:
        return self.time < other.time

class ExecutionProfile:
    """Models computational characteristics and scaling behavior of HPC workloads."""
    def __init__(self, base_time: float, min_nodes: int, max_nodes: int,
                 parallel_fraction: float = 0.95, comm_factor: float = 0.1):
        self.base_time = base_time
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.parallel_fraction = parallel_fraction
        self.comm_factor = comm_factor
        self.efficiency_threshold = 0.65
        self.historical_data: List[Tuple[float, float]] = []
        self.model_uncertainty = 0.1
    
    def update_model(self, predicted_time: float, actual_time: float) -> None:
        """Update performance model based on observed execution times."""
        self.historical_data.append((predicted_time, actual_time))
        if len(self.historical_data) > 100:
            # Maintain rolling window of historical data
            self.historical_data = self.historical_data[-100:]



    def find_optimal_allocation(self, available_nodes: int) -> Tuple[int, float, float]:
        """Determine optimal node count maximizing efficiency within constraints using advanced performance modeling."""
        best_nodes = self.min_nodes
        best_efficiency = 0
        best_time = float('inf')
        
        # Establish search boundaries for node allocation
        lower_bound = max(self.min_nodes, int(available_nodes * 0.1))  # Minimum 10% utilization
        upper_bound = min(self.max_nodes, available_nodes)
        
        try:
            # Implementation of advanced binary search for optimal allocation
            search_space = range(lower_bound, upper_bound + 1)
            
            for nodes in search_space:
                # Compute execution time and efficiency with enhanced performance model
                runtime = self.base_time * (self.min_nodes / nodes) * (1 + self.comm_factor * np.log2(nodes/self.min_nodes))
                theoretical_speedup = nodes / self.min_nodes
                actual_speedup = self.base_time / runtime
                efficiency = actual_speedup / theoretical_speedup
                
                # Update optimal allocation based on efficiency constraints
                if efficiency >= self.efficiency_threshold and runtime < best_time:
                    best_nodes = nodes
                    best_efficiency = efficiency
                    best_time = runtime
                    
            if best_time == float('inf'):
                logging.warning(f"No allocation satisfying efficiency threshold {self.efficiency_threshold}")
                # Fall back to minimum nodes if no optimal solution found
                runtime = self.base_time * (self.min_nodes / lower_bound)
                efficiency = 1.0  # Assume baseline efficiency
                return lower_bound, runtime, efficiency
                
            return best_nodes, best_time, best_efficiency
            
        except Exception as e:
            logging.error(f"Error in performance optimization: {str(e)}")
            # Provide fallback allocation strategy
            runtime = self.base_time * (self.min_nodes / lower_bound)
            efficiency = 1.0
            return lower_bound, runtime, efficiency

class AdaptiveJob:
    """Represents an HPC workload with flexible resource requirements."""
    def __init__(self, job_id: int, description: str, execution_profile: ExecutionProfile):
        self.job_id = job_id
        self.description = description
        self.execution_profile = execution_profile
        self.assigned_nodes: Optional[int] = None
        self.predicted_runtime: Optional[float] = None
        self.actual_runtime: Optional[float] = None
        self.efficiency: Optional[float] = None
        self.start_time: Optional[float] = None
        self.completion_time: Optional[float] = None

class ResourceAgent:
    """Manages computational resources with adaptive allocation strategies."""
    def __init__(self, agent_id: int, num_nodes: int):
        self.agent_id = agent_id
        self.num_nodes = num_nodes
        self.available_nodes = num_nodes
        self.assigned_jobs: List[AdaptiveJob] = []
        self.completed_jobs: List[AdaptiveJob] = []
        self.load_threshold = 0.8
        self.min_allocation = 0.2

    def calculate_load(self) -> float:
        """Compute current system load metrics."""
        return 1 - (self.available_nodes / self.num_nodes)

    def can_accommodate_job(self, job: AdaptiveJob) -> bool:
        """Check if job's minimum requirements can be met."""
        return self.available_nodes >= job.execution_profile.min_nodes
    

    def optimize_allocation(self, job: AdaptiveJob, current_time: float) -> Tuple[bool, Optional[float]]:
        """Optimize resource allocation based on system state and job requirements."""
        if not self.can_accommodate_job(job):
            return False, None

        current_load = self.calculate_load()
        available_nodes = self.available_nodes
        
        # Apply load-based allocation constraints
        if current_load > self.load_threshold:
            available_nodes = int(available_nodes * 0.8)  # Conservative allocation
            
        try:
            optimal_nodes, predicted_time, efficiency = \
                job.execution_profile.find_optimal_allocation(available_nodes)
                
            if optimal_nodes > 0 and efficiency >= job.execution_profile.efficiency_threshold:
                job.assigned_nodes = optimal_nodes
                job.predicted_runtime = predicted_time
                job.efficiency = efficiency
                job.start_time = current_time
                
                self.assigned_jobs.append(job)
                self.available_nodes -= optimal_nodes
                
                logging.info(f"Agent {self.agent_id} allocated {optimal_nodes} nodes to job {job.job_id} "
                           f"(efficiency: {efficiency:.2f}, predicted runtime: {predicted_time:.2f})")
                return True, predicted_time
                
        except ValueError as e:
            logging.warning(f"Allocation failed for job {job.job_id}: {str(e)}")
            
        return False, None

    def complete_job(self, job: AdaptiveJob, current_time: float) -> None:
        """Process job completion and resource release."""
        if job in self.assigned_jobs:
            self.assigned_jobs.remove(job)
            self.completed_jobs.append(job)
            self.available_nodes += job.assigned_nodes
            
            job.completion_time = current_time
            job.actual_runtime = current_time - job.start_time
            
            # Update performance model with actual runtime
            job.execution_profile.update_model(job.predicted_runtime, job.actual_runtime)
            
            logging.info(f"Agent {self.agent_id} completed job {job.job_id} "
                        f"(actual runtime: {job.actual_runtime:.2f}, "
                        f"predicted: {job.predicted_runtime:.2f})")



def generate_workload(num_jobs: int, resource_configs: List[Dict[str, int]], 
                     seed: Optional[int] = None) -> List[Dict]:
    """
    Generate synthetic HPC workload with sophisticated resource requirements aligned with system topology.
    
    Parameters:
        num_jobs: Number of jobs to generate
        resource_configs: List of resource agent configurations containing node counts
        seed: Random seed for reproducibility
        
    Returns:
        List of job configurations with adaptive resource requirements calibrated to system capacity
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Enhanced workload templates with system-aware parameters
    workload_templates = {
        "Data Analytics": {
            "base_time_range": (50, 150),
            "node_ratio_range": (0.1, 0.3),
            "parallel_efficiency": 0.85,
            "comm_intensity": 0.15,
            "scaling_factor": 0.9  # Efficiency scaling with node count
        },
        "Physics Simulation": {
            "base_time_range": (200, 500),
            "node_ratio_range": (0.2, 0.5),
            "parallel_efficiency": 0.90,
            "comm_intensity": 0.20,
            "scaling_factor": 0.95
        },
        "Machine Learning": {
            "base_time_range": (100, 300),
            "node_ratio_range": (0.15, 0.4),
            "parallel_efficiency": 0.80,
            "comm_intensity": 0.10,
            "scaling_factor": 0.85
        },
        "Climate Modeling": {
            "base_time_range": (300, 600),
            "node_ratio_range": (0.3, 0.6),
            "parallel_efficiency": 0.88,
            "comm_intensity": 0.25,
            "scaling_factor": 0.92
        }
    }
    
    # Calculate total system capacity from resource configurations
    total_system_nodes = sum(config["num_nodes"] for config in resource_configs)
    
    # Calculate per-agent node distributions for topology-aware allocation
    agent_node_distributions = [
        config["num_nodes"] / total_system_nodes 
        for config in resource_configs
    ]
    
    jobs = []
    for job_id in range(1, num_jobs + 1):
        # Select workload category with topology consideration
        category = random.choice(list(workload_templates.keys()))
        template = workload_templates[category]
        
        # Generate base execution time with system-scale consideration
        base_time = random.uniform(*template["base_time_range"])
        
        # Calculate node requirements based on system topology
        min_ratio, max_ratio = template["node_ratio_range"]
        
        # Adjust ratios based on system scale and topology
        adjusted_min_ratio = min_ratio * min(1.0, len(resource_configs) * 0.8)
        adjusted_max_ratio = max_ratio * min(1.0, len(resource_configs) * 0.9)
        
        # Calculate node bounds with system awareness
        min_nodes = max(4, int(total_system_nodes * adjusted_min_ratio))
        max_nodes = min(total_system_nodes, int(total_system_nodes * adjusted_max_ratio))
        
        # Ensure viable node range with topology consideration
        if min_nodes >= max_nodes:
            min_nodes = max(4, max_nodes // 2)
            
        # Calculate communication costs based on system topology
        comm_factor = template["comm_intensity"] * (1 + 0.1 * len(resource_configs))
        
        # Generate job description with enhanced metadata
        description = f"{category} Task {job_id}"
        
        # Construct job configuration with topology-aware parameters
        job = {
            "job_id": job_id,
            "description": description,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "base_time": base_time,
            "category": category,
            "parallel_efficiency": template["parallel_efficiency"],
            "comm_intensity": comm_factor,
            "scaling_factor": template["scaling_factor"],
            "topology_distribution": agent_node_distributions.copy(),
            "system_scale": total_system_nodes
        }
        
        # Log job generation with system awareness
        logging.info(f"Generated job {job_id}: {description} "
                    f"(Nodes: {min_nodes}-{max_nodes}, "
                    f"Base Time: {base_time:.2f}, "
                    f"System Scale: {total_system_nodes})")
        
        jobs.append(job)
    
    return jobs


class LLMResourceManager:
    """Manages LLM-based decision making for resource allocation."""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model =  "mistral" #"llama3.2"
        self._client = None
        self.timeout = 60.0  # Overall timeout for the request

    @property
    def client(self):
        if self._client is None:
            # Increase client timeout settings
            timeout_settings = httpx.Timeout(
                connect=10.0,    # Connection timeout
                read=40.0,      # Read timeout
                write=10.0,     # Write timeout
                pool=10.0       # Pool timeout
            )
            self._client = httpx.AsyncClient(timeout=timeout_settings)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_response(self, prompt: str) -> str:
        """Generate response from Ollama Mistral model with timeout."""
        print("\n=== LLM Request Started ===")
        try:
            async with asyncio.timeout(self.timeout):
                print(f"Sending request to {self.base_url}/api/generate...")
                # print(f"Prompt ----> {prompt}",end="\n\n")

                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                # print(f"Response status: {response.status_code}")
                
                response.raise_for_status()
                result = response.json()["response"]
                print("Response received successfully")
                return result

        except httpx.ReadTimeout:
            print("ERROR: Read timeout - Ollama is taking too long to respond")
            print("Please check if Ollama is running and the model is loaded")
            return ""
        except httpx.ConnectTimeout:
            print("ERROR: Connection timeout - Cannot connect to Ollama")
            print("Please check if Ollama is running at", self.base_url)
            return ""
        except httpx.ConnectError as e:
            print(f"ERROR: Connection failed - {str(e)}")
            print("Please check if Ollama is running at", self.base_url)
            return ""
        except Exception as e:
            print(f"ERROR: LLM request failed - {str(e)}")
            print(f"Exception type: {type(e)}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.content}")
            return ""

    async def optimize_allocation(self, job: AdaptiveJob, agent: ResourceAgent) -> Tuple[int, float]:
        """Use LLM to optimize resource allocation based on job and system characteristics."""
        system_state = {
            "total_nodes": agent.num_nodes,
            "available_nodes": agent.available_nodes,
            "current_load": agent.calculate_load(),
            "running_jobs": len(agent.assigned_jobs),
            # "job_requirements": {
            #     "min_nodes": job.execution_profile.min_nodes,
            #     "max_nodes": job.execution_profile.max_nodes,
            #     "base_time": job.execution_profile.base_time
            # }
            
            "min_nodes": job.execution_profile.min_nodes,
            "max_nodes": job.execution_profile.max_nodes,
            "base_time": job.execution_profile.base_time
            
        }

         

        prompt = f"""Given the following HPC system state and job requirements, recommend optimal node allocation in only numbers and strategies for performance scaling by providing expression for runtime estimation:  
System State: {json.dumps(system_state, indent=2)}  
  
Consider:  
1. Current system load and availability  
2. Job's resource requirements  
3. Expected performance scaling  
4. System efficiency targets  
  
Provide the recommendation in JSON format with the following structure:  
  
{{ 
  "node_allocation": "<python_integer>",  
  "runtime_estimation_formula": "<python_expression>"  
}}  
  
**Example:**  
```json  
{{  
  "node_allocation": 10,  
  "runtime_estimation_formula": "base_time * allocated_nodes"  
}}  
```  
"""
# Provide recommendation as: <nodes>,<expected_runtime>
        # print('system state and job req. --> ',  {json.dumps(system_state, indent=2)})
        response = await self.generate_response(prompt)
        print('nodes and runtime LLM response is --> ',response)

        try:
            # nodes_str, runtime_str = response.strip().split(",") # response has not parsed
            nodes_str, runtime_str = extract_metrics(response,system_state=system_state)
            print('The parsed LLM response --> ',nodes_str, runtime_str,'\n')
            # print('Here is the suggested nodes and rutime --> ',nodes_str, runtime_str)
          
            return int(nodes_str), float(runtime_str)
        
        except (ValueError, TypeError, AttributeError):
            # Fallback to baseline allocation if LLM response parsing fails
            print("No parsing happening --> ",job.execution_profile.min_nodes, job.execution_profile.base_time,'\t\t')
            return job.execution_profile.min_nodes, job.execution_profile.base_time # by default considers min_nodes and base_time

    async def prioritize_jobs(self, pending_jobs: List[AdaptiveJob], system_state: Dict) -> List[AdaptiveJob]:
        """Use LLM to prioritize pending jobs based on system state and job characteristics."""
        if not pending_jobs:
            return []
            
        jobs_info = [{
            "job_id": job.job_id,
            "description": job.description,
            "min_nodes": job.execution_profile.min_nodes,
            "max_nodes": job.execution_profile.max_nodes,
            "base_time": job.execution_profile.base_time
        } for job in pending_jobs]

        prompt = f"""Given the following pending jobs and system state, provide optimal job execution order:
Jobs: {json.dumps(jobs_info, indent=2)}
System State: {json.dumps(system_state, indent=2)}
Consider:
1. Job resource requirements
2. System utilization
3. Expected job duration
4. Fair share scheduling
Provide ordered list of job IDs."""

        response = await self.generate_response(prompt)
        print("prompt returns shceduling -------> ",prompt)
        print("Shceduling response -------->",response)
        
        try:
            ordered_ids = [int(id_str) for id_str in response.strip().split(",")]
            return sorted(pending_jobs, key=lambda job: ordered_ids.index(job.job_id))
        except:
            return pending_jobs

class AdvancedSchedulerAgent:  
    """Enhanced scheduler with combined LLM-based and asynchronous decision making."""  
    def __init__(self, agent_configs: List[Dict[str, int]]):  
        self.agents = [ResourceAgent(i, config["num_nodes"])   
                      for i, config in enumerate(agent_configs)]  
        self.event_queue: List[Event] = []  
        self.current_time = 0.0  
        self.completed_jobs: List[AdaptiveJob] = []  
        self.event_log: List[Dict] = []  
        self.llm_manager = LLMResourceManager()  
        self.pending_jobs: List[AdaptiveJob] = []  
        
        health_logger = logging.getLogger('health_monitor')
        self.health_agent = SystemHealthCheckAgent(self, logger=health_logger) # Initialize health agent 
        self._log_event("scheduler_initialization", None, None)  
  
    def _log_event(self, event_type: str, job: Optional[AdaptiveJob],   
                   agent_id: Optional[int]) -> None:  
        """Record detailed event information with temporal characteristics."""  
        event_data = {  
            "simulation_time": self.current_time,  
            "event_type": event_type,  
            "system_time": datetime.now().isoformat(),  
            "job_id": job.job_id if job else None,  
            "agent_id": agent_id,  
            "available_resources": {  
                agent.agent_id: agent.available_nodes   
                for agent in self.agents  
            }  
        }  
          
        if job:  
            event_data.update({  
                "job_description": job.description,  
                "assigned_nodes": job.assigned_nodes,  
                "predicted_runtime": job.predicted_runtime  
            })  
              
        self.event_log.append(event_data)  
        logging.info(f"Time {self.current_time:.3f}: {event_type} - " +   
                    f"Job {job.job_id if job else 'N/A'} " +  
                    f"Agent {agent_id if agent_id is not None else 'N/A'}")  
  
    def submit_job(self, job_id: int, description: str, min_nodes: int,   
                  max_nodes: int, base_time: float) -> None:  
        """Submit job with temporal tracking."""  
        profile = ExecutionProfile(base_time, min_nodes, max_nodes)  
        job = AdaptiveJob(job_id, description, profile)  
          
        event = Event(self.current_time, "job_submission", job=job)  
        heapq.heappush(self.event_queue, event)  
          
        self._log_event("job_submission", job, None)  
  
    def schedule_event(self, time: float, action: str,   
                      agent_id: Optional[int] = None,   
                      job: Optional[AdaptiveJob] = None) -> None:  
        """Schedule future event in simulation timeline."""  
        event = Event(time, action, agent_id, job)  
        heapq.heappush(self.event_queue, event)  
  
    def generate_dynamic_job(self, job_id_counter: int, total_system_nodes: int) -> AdaptiveJob:  
        """Generate a random job dynamically during the simulation."""  
        categories = {  
            "Data Analytics": {"base_time_range": (50, 150), "node_ratio_range": (0.1, 0.3)},  
            "Physics Simulation": {"base_time_range": (200, 500), "node_ratio_range": (0.2, 0.5)},  
            "Machine Learning": {"base_time_range": (100, 300), "node_ratio_range": (0.15, 0.4)},  
            "Climate Modeling": {"base_time_range": (300, 600), "node_ratio_range": (0.3, 0.6)},  
        }  
          
        category = random.choice(list(categories.keys()))  
        template = categories[category]  
  
        base_time = random.uniform(*template["base_time_range"])  
        min_nodes = max(4, int(total_system_nodes * template["node_ratio_range"][0]))  
        max_nodes = min(total_system_nodes, int(total_system_nodes * template["node_ratio_range"][1]))  
  
        description = f"{category} Dynamic Task {job_id_counter}"  
        profile = ExecutionProfile(base_time, min_nodes, max_nodes)  
          
        job = AdaptiveJob(job_id_counter, description, profile)  
        logging.info(f"Generated dynamic job {job_id_counter}: {description} "  
                    f"(Nodes: {min_nodes}-{max_nodes}, Base Time: {base_time:.2f})")  
        return job  
  
    async def _handle_job_submission_async(self, job: AdaptiveJob) -> None:  
        """Process job submission with LLM-based optimization."""  
        self.pending_jobs.append(job)  
          
        system_state = {  
            "agents": [{  
                "id": agent.agent_id,  
                "total_nodes": agent.num_nodes,  
                "available_nodes": agent.available_nodes,  
                "current_load": agent.calculate_load()  
            } for agent in self.agents]  
        }  
  
        prioritized_jobs = await self.llm_manager.prioritize_jobs(  
            self.pending_jobs, system_state  
        )  
        print("Prioritized jobs --> ", [job.job_id for job in prioritized_jobs])  
  
        for job in prioritized_jobs:  
            assigned = False  
            for agent in self.agents:  
                if agent.can_accommodate_job(job):  
                    optimal_nodes, predicted_runtime = await self.llm_manager.optimize_allocation(  
                        job, agent  
                    )  
  
                    if optimal_nodes > 0:  
                        job.assigned_nodes = optimal_nodes  
                        job.predicted_runtime = predicted_runtime  
                        job.start_time = self.current_time  
  
                        agent.assigned_jobs.append(job)  
                        agent.available_nodes -= optimal_nodes  
  
                        self.schedule_event(  
                            self.current_time + predicted_runtime,  
                            "job_completion",  
                            agent.agent_id,  
                            job  
                        )  
                        self._log_event("job_allocation", job, agent.agent_id)  
                        assigned = True  
                        self.pending_jobs.remove(job)  
                        print(f"Job {job.job_id} allocated to Agent {agent.agent_id}")  
                        break  
  
            if not assigned:  
                # Keep job in pending queue for next scheduling cycle  
                print(f"Job {job.job_id} could not be allocated and remains pending.")  
  
    async def _handle_job_completion(self, agent_id: int, job: AdaptiveJob) -> None:  
        """Process job completion with temporal tracking."""  
        agent = self.agents[agent_id]  
        agent.complete_job(job, self.current_time)  
        self.completed_jobs.append(job)  
        self._log_event("job_completion", job, agent_id)  
  
    async def run_async(self, dynamic_job_interval: float = 10.0, max_dynamic_jobs: int = 10) -> None:  
        """Async version of the simulation runner."""  
        logging.info("Starting async simulation run...")  
        job_id_counter = len(self.event_queue) + 1  
        dynamic_job_count = 0  
  
        while self.event_queue:  
            event = heapq.heappop(self.event_queue)  
            self.current_time = event.time  
  
            if event.action == "job_submission":  
                await self._handle_job_submission_async(event.job)  
            elif event.action == "job_completion":  
                await self._handle_job_completion(event.agent_id, event.job)  
  
            # Generate dynamic jobs if needed  
            if dynamic_job_count < max_dynamic_jobs and self.current_time >= dynamic_job_count * dynamic_job_interval:  
                dynamic_job = self.generate_dynamic_job(  
                    job_id_counter,   
                    sum(agent.num_nodes for agent in self.agents)  
                )  
                self.submit_job(  
                    dynamic_job.job_id,  
                    dynamic_job.description,  
                    dynamic_job.execution_profile.min_nodes,  
                    dynamic_job.execution_profile.max_nodes,  
                    dynamic_job.execution_profile.base_time  
                )  
                job_id_counter += 1  
                dynamic_job_count += 1  
  
        # Clean up  
        await self.llm_manager.close()  
        logging.info("Simulation completed")  
  
    async def run_simulation(self, dynamic_job_interval: float = 10.0, max_dynamic_jobs: int = 10) -> None:  
        """  
        Entry point to run the simulation asynchronously.  
        """  
        
        await self.health_agent.start()
        print("health agent started?    ---->", self.health_agent.running)
        await self.run_async(dynamic_job_interval, max_dynamic_jobs)  
        await self.health_agent.stop()


async def async_main():
    """Async entry point for the simulation."""
    # System configuration
    resource_configs = [
        {"num_nodes": 1024},
        {"num_nodes": 1024},
        {"num_nodes": 512}
    ]
    
    # Initialize scheduler
    scheduler = AdvancedSchedulerAgent(resource_configs)
    
    # Generate and submit initial workload
    jobs = generate_workload(10, resource_configs, seed=42)
    # health_logger.info("Health monitor initialized successfully.")  

    # Submit jobs synchronously to avoid potential race conditions
    for job in jobs:
        scheduler.submit_job(
            job["job_id"],
            job["description"],
            job["min_nodes"],
            job["max_nodes"],
            job["base_time"]
        )
    
    # Run simulation
    await scheduler.run_simulation(dynamic_job_interval=15.0, max_dynamic_jobs=2)

def main():
    """Synchronous entry point."""
    print('Starting simulation...')
    asyncio.run(async_main())

if __name__ == "__main__":
    main()