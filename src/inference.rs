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
pub struct InferenceState {
    /// Current active node indexes
    active_nodes: Vec<Option<usize>>,
}

impl InferenceState {
    pub fn new() -> Self {
        Self {
            active_nodes: Vec::new(),
        }
    }

    pub fn add(&mut self, index: usize) {
        self.active_nodes.push(Some(index));
    }

    pub fn remove(&mut self, index: usize) {
        self.active_nodes.retain(|&i| i != Some(index));
    }

    pub fn clear(&mut self) {
        self.active_nodes.clear();
    }

    fn initialize_think_inputs(&mut self, tree_root_size: usize) {
        for _ in 0..tree_root_size {
            self.active_nodes.push(Some(0));
        }
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
    graph: Graph,
    client: Client<OpenAIConfig>,
    /// Record the inference state
    state: InferenceState,
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
            graph: Graph::new(), 
            client,
            query,
            model,
            state: InferenceState::new(),
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
    fn create_json(&self, prompt: String, branch: Option<usize>) -> Result<CreateChatCompletionResponse, Error> {
        // Store the messages, which are the other nodes of this branch
        let mut context = Vec::new();
        
        // Add the prompt to the context
        context.push(ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into());
        
        if let Some(branch) = branch {
            // Add the previous nodes to the context, if any
            let nodes_indexes: Vec<usize> = self.graph.traverse(branch);
            for node_index in nodes_indexes {
                if let Some(node) = self.graph.access_node(node_index) {
                    context.push(
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(node.access_thought())
                            .build()?
                            .into()
                    );
                }
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
        branch: Option<usize>
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
    fn think(&self, branch: Option<usize>) -> Result<Thought, Error> {
        // Get the thinking prompt
        let thinking_prompt: String = if let Some(instruction) = self.access_instruction(PromptType::Thought) {
            instruction.to_string()
        } else {
            return Err(anyhow!("Think instruction is not found!"));
        };
        
        let response: CreateChatCompletionResponse = self.create_json(
            thinking_prompt, branch
        )?;
        
        Ok(
            serde_json::from_str(
                &response.choices[0].message.content.as_ref().unwrap()
            )?
        )
    }
    
    /// Run the inference, return a final thinking chain
    /// for final answer generations
    pub fn run(&mut self, parameters: InferenceParameters) -> Result<Graph, Error> {
        // Fire up N thinking threads based on `tree_root_size`
        self.state.initialize_think_inputs(
            parameters.tree_root_size
        );
        
        loop {
            // Check if there is only one active branch left
            // Return if so.
            if self.state.active_nodes.len() == 1 {
                if let Some(branch) = self.graph.get_only_one_not_pruned()? {
                    return Ok(branch);
                }
            }

            // LLM thinks the next steps/nodes
            let results: Vec<
                (
                    Option<usize>,
                    Result<Thought, Error>, 
                    Result<ClosenessToAnswer, Error>
                )
            > = self.state.active_nodes.par_iter()
                .map(
                    |input| {
                        // Create a thinking step
                        let thought = self.think(*input);
                        // LLM evaluates the closeness to the answer
                        let closeness = self.evaluate_closeness_to_answer(*input);

                        (*input, thought, closeness)
                    }
                )
                .collect();
            
            for (input_index, thought, closeness) in results {
                let thought: Thought = match thought {
                    Ok(result) => result,
                    Err(e) => return Err(e),
                };

                let closeness: ClosenessToAnswer = match closeness {
                    Ok(result) => result,
                    Err(e) => return Err(e),
                };

                // Create a new node in the graph
                let index: usize = self.graph
                    .new_node(closeness.clone(), thought);
                
                // Prune the steps/nodes that are not relevant
                if f32::from(closeness) < parameters.access_prune_threshold() {
                    self.state.remove(index);
                    self.graph.prune_branch(index);
                    
                    continue;
                }
                
                // Update the Graph
                if let Some(previous_index) = input_index {
                    self.graph.link_nodes(
                        previous_index, 
                        index
                    );
                }
            }
        }
    }
}