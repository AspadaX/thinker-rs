use std::{ops::Range, sync::{Arc, RwLock}};

use anyhow::{anyhow, Error, Result};
use async_openai::{config::OpenAIConfig, types::{ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse, ResponseFormat}, Client};
use console::Term;
use log::info;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use tokio::runtime::Runtime;
use rand::{rng, Rng};

use crate::{graph::Graph, node::Node, prompt::{ClosenessToAnswer, Instruction, PromptType, Thought}};

/// A struct that encapsulates the result of the inference process.
#[derive(Debug, Deserialize, Serialize)]
pub struct InferenceResult {
    nodes: Vec<Node>,
}

impl InferenceResult {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }
    
    pub fn get_thoughts_in_a_string(&self) -> String {
        let mut thoughts = String::new();
        for node in self.nodes.iter() {
            thoughts.push_str(node.access_thought());
        }
        
        thoughts
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParameters {
    /// The initial size of the tree. Larger indicates more
    /// possibilities, but also more computation costs.
    tree_root_size: usize,
    /// The prune threshold for the tree. Larger indicates more
    /// possibilities, but also more computation costs.
    prune_threshold: f32,
    /// Maximum depth that the model can reach. Leave `None` for unlimited depth.
    max_depth: Option<usize>,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            tree_root_size: 10,
            prune_threshold: 6.0,
            max_depth: None
        }
    }
}

impl InferenceParameters {
    pub fn new(
        tree_root_size: usize, 
        prune_threshold: f32, 
        max_depth: Option<usize>
    ) -> Self {
        Self {
            tree_root_size,
            prune_threshold,
            max_depth
        }
    }

    pub fn get_tree_root_size(&self) -> usize {
        self.tree_root_size
    }

    pub fn get_prune_threshold(&self) -> f32 {
        self.prune_threshold
    }
    
    pub fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }
}

/// LLM inference with tree of thoughts
#[derive(Debug)]
pub struct Inference {
    query: String,
    model: String,
    instructions: Vec<Instruction>,
    graph: Arc<RwLock<Graph>>,
    client: Client<OpenAIConfig>,
    runtime: Runtime,
}

impl Inference {
    pub fn new(
        api_url: String, 
        api_key: String,
        instructions: Vec<Instruction>,
        query: String,
        model: String,
    ) -> Self {
        let client: Client<OpenAIConfig> = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(api_url)
                .with_api_key(api_key)
        );

        Self {
            instructions,
            graph: Arc::new(RwLock::new(Graph::new())), 
            client,
            query,
            model,
            runtime: Runtime::new().unwrap(),
        }
    }
    
    /// Find the instruction for the given prompt type
    fn access_instruction(&self, prompt_type: PromptType) -> Option<&Instruction> {
        self.instructions.iter()
            .find(
                |instruction| 
                *instruction.access_prompt_type() == prompt_type
            )
    }
    
    /// Synchronous wrapper for generating a chat completion response
    fn generate(
        &self, 
        request: CreateChatCompletionRequest
    ) -> Result<CreateChatCompletionResponse, Error> {
        let result: CreateChatCompletionResponse = self.runtime.block_on(
            async move {
                let response: CreateChatCompletionResponse = self.client
                    .chat()
                    .create(request)
                    .await?;
                
                Ok::<CreateChatCompletionResponse, Error>(response)
            }
        )?;
        
        Ok(result)
    }
    
    /// Get a random seed for the inference session
    fn get_seed(&self) -> u32 {
        let mut random: rand::prelude::ThreadRng = rng();
        
        random.random()
    }
    
    /// Create a structural output that is ready for access 
    /// This will use the previous nodes as part of the prompt
    fn create_json(&self, prompt: String, branch: usize) -> Result<CreateChatCompletionResponse, Error> {
        // Store the messages, which are the other nodes of this branch
        let mut context: Vec<ChatCompletionRequestMessage> = Vec::new();

        // Use the previous nodes as part of the prompt, if the branch
        // Add the previous nodes to the context, if any
        let nodes_indexes: Vec<usize> = self.graph.read().unwrap().traverse_reverse(branch);
        for node_index in nodes_indexes {
            if let Some(node) = self.graph.write().unwrap().access_node(node_index) {
                context.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(node.access_thought())
                        .build()?
                        .into()
                );
            }
        }
        
        // Add the prompt to the context
        context.push(ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into());
        info!("{:?}", &context);

        // Create message for sending to the LLM
        let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
            .seed(self.get_seed())
            .model(&self.model)
            .response_format(ResponseFormat::JsonObject)
            .messages(context)
            .build()?;

        let response: CreateChatCompletionResponse = self.generate(request)?;

        Ok(response)
    }
    
    /// Evaluate how close the thinking step is to the final answer
    fn evaluate_closeness_to_answer(
        &self, 
        branch: usize
    ) -> Result<ClosenessToAnswer, Error> {
        // Get the closeness prompt
        let closeness_prompt: String = if let Some(instruction) = self.access_instruction(PromptType::ClosenessToAnswer) {
            instruction.to_string()
        } else {
            return Err(anyhow!("ClosenessToAnswer instruction is not found!"));
        };

        loop {
            let response: CreateChatCompletionResponse = self.create_json(closeness_prompt.clone(), branch)?;
            
            match serde_json::from_str(
                &response.choices[0].message.content.as_ref().unwrap()
            ) {
                Ok(result) => return Ok(result),
                Err(_) => {
                    continue;
                }
            };
        }
    }

    /// Generate a thinking step with the LLM
    /// A None value in the `branch` indicates the first node of the graph.
    fn think(&self, branch: usize) -> Result<Thought, Error> {
        // Get the thinking prompt
        let thinking_prompt: String = if let Some(instruction) = self.access_instruction(PromptType::Thought) {
            instruction.to_string()
        } else {
            return Err(anyhow!("Think instruction is not found!"));
        };

        loop {
            let response: CreateChatCompletionResponse = self.create_json(
                format!(
                    "User Query:\n{} \n{}", self.query, thinking_prompt
                ), branch
            )?;

            match serde_json::from_str(
                &response.choices[0].message.content.as_ref().unwrap()
            ) {
                Ok(thought) => {
                    return Ok(thought);
                },
                Err(_) => {
                    continue;
                }
            };
        }
    }
    
    fn think_in_parallel(
        &self, 
        branch: usize, 
        tree_root_size: usize
    ) -> Result<Vec<(usize, Thought)>, Error> {
        let nodes_range: std::ops::Range<usize> = 0..tree_root_size;
        log::debug!("Starting think_in_parallel for branch: {}, tree_root_size: {}", branch, tree_root_size);

        // LLM thinks the next steps/nodes
        let results: Vec<(usize, Result<Thought, Error>)> = nodes_range.into_par_iter()
            .map(
                |_| {
                    log::debug!("Thinking for branch: {}", branch);
                    // Create nodes for N different thinking steps
                    // this is based on the `tree_root_size`
                    let thought: Result<Thought, Error> = self.think(branch);

                    (branch, thought)
                }
            )
            .collect();

        let mut final_results: Vec<(usize, Thought)> = Vec::new();
        for result in results {
            let thought: Thought = result.1?;
            log::debug!("Generated thought for branch: {}", result.0);
            final_results.push(
                (result.0, thought)
            );
        }

        log::debug!("Completed think_in_parallel for branch: {}", branch);
        Ok(final_results)
    }
    
    /// Call this method before starting inferencing. It will create root nodes
    /// for the session. 
    fn initialize_root_nodes(&self, root_nodes_size: Range<usize>) -> Result<(), Error> {
        // Collect the results of thinking in parallel for each active node
        let results: Vec<Result<Vec<(usize, Thought)>, Error>> = root_nodes_size
            .clone()
            .into_par_iter()
            .map(
                |branch| {
                    self.think_in_parallel(
                        branch, 
                        root_nodes_size.end
                    )
                }
            )
            .collect();

        // Add the new thoughts as nodes into the graph
        // A None in `input_index` means that the thought is a root node.
        for nodes_info in results {
            let nodes_info = nodes_info?;
            for (_, thought) in nodes_info {
                let index: usize = self.graph.write().unwrap().new_node(thought);

                let closeness: ClosenessToAnswer = self.evaluate_closeness_to_answer(index)?;

                // Update the node's closeness to the answer
                if let Some(node) = self.graph.write().unwrap().access_node(index) {
                    node.update_closeness(closeness.clone().into());
                }
            }
        }

        Ok(())
    } 
    
    /// Run the inference, return a final thinking chain
    /// for final answer generations
    pub fn run(&mut self, parameters: InferenceParameters) -> Result<Graph, Error> {
        let terminal = Term::stdout();
        let root_nodes: Range<usize> = 0..parameters.tree_root_size;
        self.initialize_root_nodes(root_nodes)?;
        terminal.write_line(
            &console::style("🌳 Starting inference tree").cyan().to_string()
        )?;

        let mut depth: usize = 1;
        let pb = indicatif::ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        loop {
            pb.set_message(format!("Depth {}: Checking branches", depth));

            if let Some(branch) = self.graph.read().unwrap().get_only_one_not_pruned()? {
                pb.finish_with_message("🎯 Found single active branch");
                terminal.write_line(
                    &console::style(format!("✨ Completed at depth {} with single branch", depth)).green().to_string()
                )?;
                return Ok(branch);
            }

            let active_nodes: Vec<usize> = self.graph.read().unwrap().get_end_nodes();
            pb.set_message(format!("Depth {}: {} active nodes", depth, active_nodes.len()));

            if Some(depth) == parameters.max_depth {
                pb.finish_with_message("📏 Max depth reached");
                terminal.write_line(
                    &console::style(format!("🧠 Choosing best node from {} candidates", active_nodes.len())).yellow().to_string()
                )?;

                let (highest_index, highest_score) = active_nodes.iter()
                    .fold((0, 0.0), |(max_idx, max_score), &idx| {
                        let score: f32 = self.graph.write().unwrap().access_node(idx)
                            .map(|n| n.get_closeness())
                            .unwrap_or(0.0);
                        if score > max_score { (idx, score) } else { (max_idx, max_score) }
                    });

                let branch = self.graph.read().unwrap()
                    .get_branch_from_end_node(highest_index)?;

                terminal.write_line(
                    &console::style(format!("🏆 Best node score: {:.2}", highest_score)).green().to_string()
                )?;
                return Ok(branch);
            }

            let processing_bar = indicatif::ProgressBar::new(active_nodes.len() as u64);
            processing_bar.set_style(indicatif::ProgressStyle::default_bar()
                .template("{spinner} [{bar:40.cyan/blue}] {pos}/{len} nodes processed")
                .unwrap()
                .progress_chars("##-"));

            let results: Vec<Result<Vec<(usize, Thought)>, Error>> = active_nodes
                .into_par_iter()
                .map(|node_index| {
                    processing_bar.inc(1);
                    self.think_in_parallel(node_index, parameters.tree_root_size)
                })
                .collect();

            processing_bar.finish_and_clear();
            terminal.write_line(
                &console::style(format!("🌱 Generated {} node groups", results.len())).dim().to_string()
            )?;

            let mut nodes_infos: Vec<Vec<(usize, Thought)>> = Vec::new();
            for nodes_info in results {
                nodes_infos.push(nodes_info?);
            }

            let total_new_nodes = nodes_infos.iter().map(|v| v.len()).sum::<usize>();
            let node_bar = indicatif::ProgressBar::new(total_new_nodes as u64);
            node_bar.set_style(indicatif::ProgressStyle::default_bar()
                .template("{spinner} [{bar:40.magenta/blue}] {pos}/{len} nodes evaluated")
                .unwrap()
                .progress_chars("##-"));

            for nodes in nodes_infos {
                for (previous_node_index, thought) in nodes {
                    let index: usize = self.graph.write().unwrap().new_node(thought);
                    node_bar.inc(1);

                    let closeness: ClosenessToAnswer = self.evaluate_closeness_to_answer(index)?;
                    if let Some(node) = self.graph.write().unwrap().access_node(index) {
                        node.update_closeness(closeness.clone().into());
                    }

                    self.graph.write().unwrap().link_nodes(previous_node_index, index);
                }
            }

            node_bar.finish_and_clear();
            terminal.write_line(
                &console::style(format!("📈 Depth {} completed with {} total nodes", depth, self.graph.read().unwrap().len())).cyan().to_string()
            )?;

            depth += 1;
        }
    }
}