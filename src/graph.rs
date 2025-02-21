use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use anyhow::{Result, Error};

use crate::node::Node;

#[derive(Debug, Serialize, Deserialize)]
pub struct Graph {
    nodes: Vec<Node>
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create a new node in the graph
    pub fn new_node(
        &mut self,
        closeness_to_answer: f32, 
        thought: String, 
        reflection: Option<String>
    ) -> usize {
        let node = Node::new(
            closeness_to_answer, 
            thought, 
            reflection
        );
        self.nodes.push(node);

        // Return the latest node's index
        self.nodes.len() - 1
    }

    /// Child node's previous node is the parent node, and vice versa
    pub fn link_nodes(&mut self, parent_node: usize, child_node: usize) {
        if let Some(node) = self.nodes.get_mut(parent_node) {
            node.assign_next_node(child_node);
        }

        if let Some(node) = self.nodes.get_mut(child_node) {
            node.assign_previous_node(parent_node);
        }
    }

    /// Check if this branch is pruned
    pub fn is_pruned(&self, branch_indexes: Vec<usize>) -> bool {
        for (index, node) in self.nodes.iter().enumerate() {
            if branch_indexes.contains(&index) {
                if node.is_pruned() {
                    return true;
                }
            }
        }

        false
    }

    /// Get all nodes on the same branch
    pub fn traverse(&self, start: usize) -> Vec<usize> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = vec![start];
        let mut branch_nodes_indexes: Vec<usize> = Vec::new();

        while let Some(node_index) = stack.pop() {
            if visited.insert(node_index) {
                branch_nodes_indexes.push(node_index);
                if let Some(node) = self.nodes.get(node_index) {
                    if let Some(next_index) = node.get_next() {
                        stack.push(next_index);
                    }
                }
            }
        }

        branch_nodes_indexes
    }

    /// Save the graph to a local position
    pub fn save(&self, path: &str) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(&self)?;
        let _ = std::fs::write(path, json);

        Ok(())
    }

    /// Load a graph from a local position
    pub fn load(path: &str) -> Result<Self, Error> {
        let json: String = std::fs::read_to_string(path)?;

        Ok(serde_json::from_str::<Graph>(&json)?)
    }
}