use std::sync::{Arc, Mutex};

use async_openai::{config::OpenAIConfig, Client};

use crate::graph::Graph;

/// LLM inference with tree of thoughts
pub struct Inference {
    graph: Arc<Mutex<Graph>>,
    client: Client<OpenAIConfig>
}

impl Inference {
    pub fn new(
        api_url: String, 
        api_key: String
    ) -> Self {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(api_url)
                .with_api_key(api_key)
        );

        Self {
            graph: Arc::new(Mutex::new(Graph::new())), 
            client: client
        }
    }

    pub fn generate(&self, query: String) -> String {}
}