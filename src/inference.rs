use std::{ops::Range, sync::{Arc, Mutex}};

use anyhow::{anyhow, Error, Result};
use async_openai::{config::OpenAIConfig, types::{ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse, ResponseFormat}, Client};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

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
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            tree_root_size: 10,
            prune_threshold: 0.5,
        }
    }
}

impl InferenceParameters {
    pub fn new(
        tree_root_size: usize, 
        prune_threshold: f32, 
    ) -> Self {
        Self {
            tree_root_size,
            prune_threshold,
        }
    }

    pub fn access_tree_root_size(&self) -> usize {
        self.tree_root_size
    }

    pub fn access_prune_threshold(&self) -> f32 {
        self.prune_threshold
    }
}

/// LLM inference with tree of thoughts
/// Step 1: get the user query. 
/// Step 2: LLM think for multiple branches to resolve the query, they are the 
/// first batch of nodes.
/// Step 3: Iterate over the nodes and generate the next batch of nodes.
/// until the LLM reaches the only branch that is feasible.
/// Step 4: Collect the user query and the branch's thoughts, 
/// throw that into a model for generating the final answer. 
pub struct Inference {
    query: String,
    model: String,
    instructions: Vec<Instruction>,
    graph: Arc<Mutex<Graph>>,
    client: Client<OpenAIConfig>,
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
            graph: Arc::new(Mutex::new(Graph::new())), 
            client,
            query,
            model,
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
        let async_runtime = tokio::runtime::Runtime::new().unwrap();
        let result: CreateChatCompletionResponse = async_runtime.block_on(
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
    
    /// Create a structural output that is ready for access 
    /// This will use the previous nodes as part of the prompt
    fn create_json(&self, prompt: String, branch: usize) -> Result<CreateChatCompletionResponse, Error> {
        let mut graph = self.graph.lock().unwrap();
        
        // Store the messages, which are the other nodes of this branch
        let mut context = Vec::new();
        
        // Add the prompt to the context
        context.push(ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into());
        
        // Use the previous nodes as part of the prompt, if the branch
        // Add the previous nodes to the context, if any
        let nodes_indexes: Vec<usize> = graph.traverse(branch);
        for node_index in nodes_indexes {
            if let Some(node) = graph.access_node(node_index) {
                context.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(node.access_thought())
                        .build()?
                        .into()
                );
            }
        }
        
        // Create message for sending to the LLM
        let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
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
        
        let response: CreateChatCompletionResponse = self.create_json(
            closeness_prompt, 
            branch
        )?;
        
        Ok(
            serde_json::from_str(
                &response.choices[0].message.content.as_ref().unwrap()
            )?
        )
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
        
        let response: CreateChatCompletionResponse = self.create_json(
            format!(
                "User Query:\n{} \n{}", self.query, thinking_prompt
            ), branch
        )?;
        
        Ok(
            serde_json::from_str(
                &response.choices[0].message.content.as_ref().unwrap()
            )?
        )
    }
    
    fn think_in_parallel(
        &self, 
        branch: usize, 
        tree_root_size: usize
    ) -> Result<Vec<(usize, Thought)>, Error> {
        let nodes_range: std::ops::Range<usize> = 0..tree_root_size;
        // LLM thinks the next steps/nodes
        let results: Vec<(usize, Result<Thought, Error>)> = nodes_range.into_par_iter()
            .map(
                |_| {
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
            final_results.push(
                (result.0, thought)
            );
        }
        
        Ok(final_results)
    }
    
    /// Call this method before starting inferencing. It will create root nodes
    /// for the session. 
    fn initialize_root_nodes(&self, root_nodes_size: Range<usize>) -> Result<(), Error> {
        let mut graph = self.graph.lock().unwrap();
        // Collect the results of thinking in parallel for each active node
        let results: Vec<Result<Vec<(usize, Thought)>, Error>> = root_nodes_size
            .clone()
            .into_par_iter()
            .map(
                |branch| 
                self.think_in_parallel(
                    branch, 
                    root_nodes_size.end
                )
            )
            .collect();
        
        // Add the new thoughts as nodes into the graph
        // A None in `input_index` means that the thought is a root node.
        for nodes_info in results {
            let nodes_info = nodes_info?;
            for (_, thought) in nodes_info {
                let index: usize = graph.new_node(thought);
                let closeness: ClosenessToAnswer = self.evaluate_closeness_to_answer(
                    index
                )?;
                
                // Update the node's closeness to the answer
                if let Some(node) = graph.access_node(index) {
                    node.update_closeness(closeness.into());
                }
            }
        }
        
        Ok(())
    } 
    
    /// Run the inference, return a final thinking chain
    /// for final answer generations
    pub fn run(&mut self, parameters: InferenceParameters) -> Result<Graph, Error> {
        let root_nodes: Range<usize> = 0..parameters.tree_root_size;
        self.initialize_root_nodes(root_nodes)?;
        
        loop {
            let mut graph = self.graph.lock().unwrap();
            
            // Check if there is only one active branch left
            // Return if so.
            if let Some(branch) = graph.get_only_one_not_pruned()? {
                return Ok(branch);
            }
            
            // Collect the results of thinking in parallel for each active node
            let results: Vec<Result<Vec<(usize, Thought)>, Error>> = graph
                .get_active_nodes()                
                .into_par_iter()
                .map(
                    |node_index| 
                    self.think_in_parallel(
                        node_index, 
                        parameters.tree_root_size
                    )
                )
                .collect();
            
            // Add the new thoughts as nodes into the graph
            // A None in `input_index` means that the thought is a root node.
            let mut nodes_infos: Vec<Vec<(usize, Thought)>> = Vec::new();
            for nodes_info in results {
                let nodes_info = nodes_info?;
                nodes_infos.push(nodes_info);
            }
            
            for nodes in nodes_infos {
                for (previous_node_index, thought) in nodes {
                    // Create a new node in the graph
                    let index: usize = graph.new_node(thought);
                    
                    let closeness: ClosenessToAnswer = self.evaluate_closeness_to_answer(index)?;
                    if let Some(node) = graph.access_node(index) {
                        node.update_closeness(closeness.into()); 
                    }
                    
                    // Link the node with the previous one
                    graph.link_nodes(previous_node_index, index);
                }
            }
        }
    }
}