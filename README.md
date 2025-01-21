# swarm-agents-llms
LLM-powered multi-agent simulation framework for resource management system

## Components
1. SystemHealthCheckAgent: This agent monitors the system's health and state in real-time (5 secs). It collects data on the scheduler's state, resource agents' usage, and job statuses, ensuring the system operates smoothly.
2. ExecutionProfile: Models the computational characteristics and scaling behavior of HPC workloads. It enables dynamic resource allocation based on performance metrics and historical data.
3. AdaptiveJob: Represents an HPC workload with flexible resource requirements. Each job includes details like assigned nodes, runtime estimations, and efficiency metrics.
4. ResourceAgent: Manages computational resources using adaptive allocation strategies. It calculates system load, determines resource availability, and optimizes job assignments to maintain optimal performance.
5. LLMResourceManager(LLM-RM): Handles language model-based decision-making for resource allocation and job prioritization. It communicates with an LLM server (local) to optimize scheduling based on system state and job requirements.
6. AdvancedShcedulerAgent: The core scheduler that integrates LLM-based and asynchronous decision-making processes
8. Logging: This implements logging for system events and health states. Logs are stored in the ```./swarm_logs/``` directory with detailed timestamps and event information.

## Configuration
* Resource Configuration: Define the number of nodes for each resource agent in the resource_configs list within async_main()
* Job Generation: Adjust the number of initial and dynamic jobs by modifying the parameters in generate_workload() and run_simluation()

## Setup
1. **Clone the repository:**
   ```sh
   git clone git@github.com:swarm-workflows/swarm-agents-llms.git
   cd swarm-agents
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running the Main Script
To start the swarm agents and monitor their activity, run:
```sh
python swarm_agents.py

