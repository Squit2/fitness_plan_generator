# How to Tell if Exercises are Being Retrieved

## 🔍 Debug Mode Feature

The improved chatbot now includes a **Debug Mode** that shows you exactly which exercises are being retrieved from the database for each query.

## How to Enable Debug Mode

1. **Open the sidebar** (left side of the app)
2. **Scroll down** to the bottom of the sidebar
3. **Check the box** labeled "🔍 Debug Mode (Show Retrieved Exercises)"
4. **Ask a question** in the chat

## What You'll See

When Debug Mode is enabled, after each AI response, you'll see:

### 1. **Debug Expander**
An expandable section appears below each response showing:
- **Number of exercises retrieved** (e.g., "🔍 Debug: Retrieved 8 Exercises")
- **List of exercise titles** that were found
- **Similarity scores** (distance values) for each exercise
- **Your original query** for reference

### Example Output:
```
🔍 Debug: Retrieved 8 Exercises
───────────────────────────────
Exercises retrieved from database:
1. **Barbell Bench Press** (distance: 0.4523)
2. **Dumbbell Flyes** (distance: 0.5124)
3. **Push-ups** (distance: 0.5234)
...
💡 Lower distance = more similar to your query: 'I want to work on my chest...'
ℹ️ These exercises were used as context for the AI response above.
```

## Understanding the Information

### Distance Scores
- **Lower distance = More similar** to your query
- **Higher distance = Less similar** to your query
- Distance is calculated using **L2 (Euclidean) distance** in the embedding space
- Typical values range from 0.3 (very similar) to 1.5+ (less similar)

### What This Tells You

1. **Retrieval is Working**: If you see exercises listed, the RAG system is successfully finding relevant exercises
2. **Relevance Quality**: Lower distance scores indicate better matches
3. **Query Understanding**: If retrieved exercises match your intent, the embedding model is working well
4. **Database Coverage**: You can see if the database has exercises relevant to your query

## Troubleshooting

### No Exercises Shown?
- Check that Debug Mode checkbox is enabled
- Verify the exercise database loaded correctly (check for errors at startup)
- Try a different query - some queries may not match any exercises

### Exercises Don't Seem Relevant?
- The embedding model might need fine-tuning
- Try rephrasing your query
- Check if the exercise database has relevant entries

### High Distance Scores?
- Scores above 1.0 may indicate less relevant matches
- Try more specific queries
- The system still uses these exercises, but they're weighted less

## Technical Details

### How It Works

1. **Query Encoding**: Your question is converted to an embedding vector
2. **FAISS Search**: The system searches the exercise database using vector similarity
3. **Top-K Retrieval**: Returns the top 8 most similar exercises (configurable)
4. **Context Building**: These exercises are formatted and sent to the AI model
5. **Response Generation**: The AI uses these exercises as context to generate a response

### Retrieval Function

The `retrieve_exercise_context()` function:
- Uses **SentenceTransformer** embeddings
- Searches using **FAISS** (Facebook AI Similarity Search)
- Returns exercises with their similarity scores
- Handles errors gracefully

## Code Location

- **Retrieval Function**: `retrieve_exercise_context()` (line ~216)
- **Debug Display**: Chat UI section (line ~564)
- **Debug Toggle**: Sidebar (line ~502)

## Tips for Best Results

1. **Be Specific**: "chest exercises" is better than "exercises"
2. **Use Keywords**: Include body parts, equipment, or goals
3. **Check Debug Mode**: Always verify retrieval is working
4. **Compare Queries**: Try different phrasings to see which retrieves better exercises

## Example Queries to Test

Try these to see retrieval in action:

- "I want to build muscle in my chest"
- "What are good leg exercises for beginners?"
- "Show me cardio workouts"
- "I need a full body workout plan"
- "What exercises help with back pain?"

Each will show different retrieved exercises based on semantic similarity!






