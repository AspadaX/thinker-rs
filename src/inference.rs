use std::sync::{Arc, Mutex};

use async_openai::{config::OpenAIConfig, Client};

use crate::graph::Graph;

/// LLM 
pub struct Inference {
    graph: Arc<Mutex<Graph>>,
    client: Client<OpenAIConfig>
}