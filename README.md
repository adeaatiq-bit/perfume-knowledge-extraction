# Perfume Domain Knowledge Graph Construction Pipeline

This is an end-to-end Neuro-Symbolic Information Extraction system designed to transform unstructured text about fragrances into a queryable Semantic Knowledge Graph.

## Project Structure
  `perfume_ontology.ttl` - The domain ontology mapping out classes (e.g., Brand, Accord, OlfactoryFamily, Perfumer) also including the logical constraints.
  `generate_dataset.py` - Includes the synthetic data generation engine using LLMs to create a tiered-complexity JSONL dataset.
  `train_ner.py` - The Named Entity Recognition (NER) training script using spaCy/Transformers.
  `verify_relations.py` - The symbolic verification layer enforcing domain/range logic onto model extractions.
 `rdf_exporter.py` - Converts valid extractions into standardized Resource Description Framework (RDF) triples.
