# Tests Folder

This folder contains test scripts for verifying the functionality of the Hotel Recommendation RAG System.

## Key Test Files

- `test_rag.py` - Basic tests for the RAG system
  - Tests query processing with location specifications
  - Tests query processing with feature specifications
  - Tests combined location and feature queries
  - Verifies correct response structure and metadata

- `test_advanced_search.py` - Tests for advanced search features
  - Tests query expansion functionality
  - Tests hybrid search combining vector and BM25 retrieval
  - Tests misspelling correction
  - Tests synonym expansion for hotel domain terms

## Running the Tests

You can run the tests using the provided run scripts:

```bash
# Run basic RAG tests
run.bat test  # On Windows
./run.sh test  # On Linux/macOS

# Run advanced search tests
run.bat test-advanced  # On Windows
./run.sh test-advanced  # On Linux/macOS
```

## Test Prerequisites

- Tests require processed data in ChromaDB
- Make sure to complete the data processing steps before running tests
- The tests use the sample dataset if full datasets are not available

## Adding New Tests

When adding new functionality to the system, please add corresponding tests:

1. Create a new test file in this directory
2. Add test cases for each aspect of the new functionality
3. Ensure tests can run independently without external dependencies
4. Update the run scripts to include your new tests

## Test Design Principles

- Tests should verify both positive and negative scenarios
- Each test should be self-contained and not depend on the state from other tests
- Tests should provide clear error messages when they fail
- Tests should run quickly to enable rapid development cycles
