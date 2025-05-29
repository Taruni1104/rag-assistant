from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Step 1: Create the embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",  # This works in the lab!
        params=embed_params,
    )
    
    return watsonx_embedding

# Step 2: Embed your query
query = "How are you?"
embedding_model = watsonx_embedding()
embedding = embedding_model.embed_query(query)

# Step 3: Print first 5 numbers from embedding
print("First 5 embedding numbers:", embedding[:5])
