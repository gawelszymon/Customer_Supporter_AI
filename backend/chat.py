def initialize_index():
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created directory: {docs_path}")
        return None
    
    documents = SimpleDirectoryReader(docs_path).load_data()
    if not documents:
        print("No documents found in the specified directory.")
        return None
        
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Initialize the index
index = initialize_index()

def chatbot(input_text):
    if index is None:
        return "Error: No documents found in the specified directory.", []

    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    
    relevant_files = []
    for node_with_score in response.source_nodes:
        file = node_with_score.node.metadata['file_name']
        full_file_path = Path(docs_path, file).resolve()
        
        if full_file_path not in relevant_files:
            relevant_files.append(full_file_path)
    
    return response.response, relevant_files

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs=[
        gr.components.Textbox(label="Response"),
        gr.components.File(label="Relevant Files")
    ],
    title="Custom-trained AI Chatbot",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=False)