import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import mongoClientPromise from '@/app/lib/mongodb';

export async function POST(req: Request) {
  try {
    // Ensure the MongoDB client is available
    const client = await mongoClientPromise;
    if (!client) throw new Error("Failed to connect to MongoDB");

    // Define the database and collection
    const dbName = "docs";
    const collectionName = "embeddings";
    const collection = client.db(dbName).collection(collectionName);

    // Read and validate the input question
    const question = await req.text();
    if (!question) return new Response("Invalid input", { status: 400 });

    // Create embeddings using OpenAI
    const embeddings = new OpenAIEmbeddings({
      modelName: 'text-embedding-ada-002',
      stripNewLines: true,
    });

    // Initialize the vector store with MongoDB
    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
      collection,
      indexName: "default",
      textKey: "text", 
      embeddingKey: "embedding",
    });

    // Configure the retriever to fetch only the most similar document
    const retriever = vectorStore.asRetriever({
      searchType: "similarity",
      k: 1  // Fetch only one most similar document
    });

    // Retrieve the most relevant document
    const retrieverOutput = await retriever.getRelevantDocuments(question);

    // Return the response as JSON
    return new Response(JSON.stringify(retrieverOutput), {
      headers: { "Content-Type": "application/json" },
    });

  } catch (error) {
    // Handle errors appropriately
    console.error("Error during the POST request:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}