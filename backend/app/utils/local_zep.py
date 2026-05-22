"""
Local Zep Client Drop-in Replacement
Decouples DealSim from the external Zep Cloud API and runs everything locally.
"""

import os
import json
import uuid
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('dealsim.local_zep')

_graph_locks = {}
_locks_lock = threading.Lock()

def _get_graph_lock(graph_id: str) -> threading.Lock:
    with _locks_lock:
        if graph_id not in _graph_locks:
            _graph_locks[graph_id] = threading.Lock()
        return _graph_locks[graph_id]


EXTRACT_PROMPT = """
You are an expert knowledge graph builder. Your task is to analyze the provided text and extract entities and relationships matching the target ontology schema.

## Target Ontology Schema:
{ontology_schema}

## Text to Analyze:
\"\"\"{text}\"\"\"

## Instructions:
1. **Entity Extraction**:
   - Extract only entities that map to the defined entity types.
   - For each entity, generate a unique name (e.g. Person name, company name).
   - Try to find the values for their attributes as defined in the ontology schema. Do not invent attributes.
   - Provide a concise `summary` of who/what they are based on the text.
   
2. **Relationship (Edge) Extraction**:
   - Extract relationships between the extracted entities.
   - The relationships MUST match the defined edge types and their allowed `source_targets` constraints.
   - For each relationship, specify:
     - `source`: The name of the source entity.
     - `target`: The name of the target entity.
     - `relationship`: The UPPER_SNAKE_CASE relationship type.
     - `fact`: A detailed natural language sentence summarizing this specific fact.
     - `attributes`: A dictionary matching the edge type's attributes.

## Output Format:
You MUST respond with a valid JSON object matching the following structure. Do NOT include markdown code fences or any other text.
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "EntityType",
      "summary": "Brief summary",
      "attributes": {{
        "attribute_name": "value"
      }}
    }}
  ],
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "relationship": "RELATIONSHIP_TYPE",
      "fact": "Fact sentence describing the relation",
      "attributes": {{
        "attribute_name": "value"
      }}
    }}
  ]
}}
"""

class LocalEpisode:
    def __init__(self, uuid_str: str, data: str, type_str: str):
        self.uuid_ = uuid_str
        self.uuid = uuid_str
        self.data = data
        self.type = type_str
        self.processed = True

class LocalNode:
    def __init__(self, uuid_str: str, name: str, labels: List[str], summary: str, attributes: Dict[str, Any], created_at: str = None):
        self.uuid_ = uuid_str
        self.uuid = uuid_str
        self.name = name
        self.labels = labels
        self.summary = summary
        self.attributes = attributes
        self.created_at = created_at or datetime.now().isoformat()

class LocalEdge:
    def __init__(self, uuid_str: str, name: str, fact: str, source_node_uuid: str, target_node_uuid: str, attributes: Dict[str, Any], created_at: str = None, valid_at: str = None, invalid_at: str = None, expired_at: str = None):
        self.uuid_ = uuid_str
        self.uuid = uuid_str
        self.name = name
        self.fact = fact
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.attributes = attributes
        self.created_at = created_at or datetime.now().isoformat()
        self.valid_at = valid_at or datetime.now().isoformat()
        self.invalid_at = invalid_at
        self.expired_at = expired_at

class LocalSearchResult:
    def __init__(self, edges: List[LocalEdge], nodes: List[LocalNode]):
        self.edges = edges
        self.nodes = nodes

class LocalEpisodeManager:
    def __init__(self, client: 'LocalZep'):
        self.client = client
        
    def get(self, uuid_: str) -> LocalEpisode:
        for graph_id in self.client._list_graph_ids():
            graph = self.client._load_graph(graph_id)
            if uuid_ in graph.get("episodes", {}):
                ep_data = graph["episodes"][uuid_]
                return LocalEpisode(uuid_, ep_data.get("data", ""), ep_data.get("type", ""))
        return LocalEpisode(uuid_, "", "")

class LocalNodeManager:
    def __init__(self, client: 'LocalZep'):
        self.client = client
        
    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str = None) -> List[LocalNode]:
        graph = self.client._load_graph(graph_id)
        nodes_dict = graph.get("nodes", {})
        nodes_list = sorted(nodes_dict.values(), key=lambda x: x.get("created_at", ""))
        
        start_idx = 0
        if uuid_cursor:
            for idx, node in enumerate(nodes_list):
                if node.get("uuid") == uuid_cursor:
                    start_idx = idx + 1
                    break
                    
        page = nodes_list[start_idx : start_idx + limit]
        return [
            LocalNode(
                uuid_str=n["uuid"],
                name=n["name"],
                labels=n["labels"],
                summary=n["summary"],
                attributes=n["attributes"],
                created_at=n.get("created_at")
            )
            for n in page
        ]
        
    def get_entity_edges(self, node_uuid: str) -> List[LocalEdge]:
        edges_list = []
        for graph_id in self.client._list_graph_ids():
            graph = self.client._load_graph(graph_id)
            for edge_uuid, edge in graph.get("edges", {}).items():
                if edge.get("source_node_uuid") == node_uuid or edge.get("target_node_uuid") == node_uuid:
                    edges_list.append(
                        LocalEdge(
                            uuid_str=edge["uuid"],
                            name=edge["name"],
                            fact=edge["fact"],
                            source_node_uuid=edge["source_node_uuid"],
                            target_node_uuid=edge["target_node_uuid"],
                            attributes=edge["attributes"],
                            created_at=edge.get("created_at"),
                            valid_at=edge.get("valid_at"),
                            invalid_at=edge.get("invalid_at"),
                            expired_at=edge.get("expired_at")
                        )
                    )
        return edges_list
        
    def get(self, uuid_: str) -> Optional[LocalNode]:
        for graph_id in self.client._list_graph_ids():
            graph = self.client._load_graph(graph_id)
            if uuid_ in graph.get("nodes", {}):
                n = graph["nodes"][uuid_]
                return LocalNode(
                    uuid_str=n["uuid"],
                    name=n["name"],
                    labels=n["labels"],
                    summary=n["summary"],
                    attributes=n["attributes"],
                    created_at=n.get("created_at")
                )
        return None

class LocalEdgeManager:
    def __init__(self, client: 'LocalZep'):
        self.client = client
        
    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str = None) -> List[LocalEdge]:
        graph = self.client._load_graph(graph_id)
        edges_dict = graph.get("edges", {})
        edges_list = sorted(edges_dict.values(), key=lambda x: x.get("created_at", ""))
        
        start_idx = 0
        if uuid_cursor:
            for idx, edge in enumerate(edges_list):
                if edge.get("uuid") == uuid_cursor:
                    start_idx = idx + 1
                    break
                    
        page = edges_list[start_idx : start_idx + limit]
        return [
            LocalEdge(
                uuid_str=e["uuid"],
                name=e["name"],
                fact=e["fact"],
                source_node_uuid=e["source_node_uuid"],
                target_node_uuid=e["target_node_uuid"],
                attributes=e["attributes"],
                created_at=e.get("created_at"),
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                expired_at=e.get("expired_at")
            )
            for e in page
        ]

class LocalGraphManager:
    def __init__(self, client: 'LocalZep'):
        self.client = client
        self.episode = LocalEpisodeManager(client)
        self.node = LocalNodeManager(client)
        self.edge = LocalEdgeManager(client)
        
    def create(self, graph_id: str, name: str, description: str):
        graph = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "ontology": {},
            "nodes": {},
            "edges": {},
            "episodes": {}
        }
        self.client._save_graph(graph_id, graph)
        logger.info(f"LocalGraph: Graph '{graph_id}' created successfully.")
        
    def delete(self, graph_id: str):
        filepath = self.client._get_filepath(graph_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"LocalGraph: Graph '{graph_id}' deleted.")
            
    def set_ontology(self, graph_ids: List[str], entities: Optional[Dict] = None, edges: Optional[Dict] = None):
        ontology_schema = {"entity_types": [], "edge_types": []}
        
        if entities:
            for entity_name, entity_class in entities.items():
                attributes = []
                fields_dict = getattr(entity_class, 'model_fields', {}) or getattr(entity_class, '__fields__', {})
                for field_name, field_obj in fields_dict.items():
                    attributes.append({
                        "name": field_name,
                        "description": getattr(field_obj, 'description', '') or ''
                    })
                ontology_schema["entity_types"].append({
                    "name": entity_name,
                    "description": entity_class.__doc__ or '',
                    "attributes": attributes
                })
                
        if edges:
            for edge_name, edge_info in edges.items():
                if isinstance(edge_info, tuple):
                    edge_class, source_targets = edge_info
                else:
                    edge_class = edge_info
                    source_targets = []
                    
                attributes = []
                fields_dict = getattr(edge_class, 'model_fields', {}) or getattr(edge_class, '__fields__', {})
                for field_name, field_obj in fields_dict.items():
                    attributes.append({
                        "name": field_name,
                        "description": getattr(field_obj, 'description', '') or ''
                    })
                    
                st_list = []
                for st in source_targets:
                    st_list.append({
                        "source": getattr(st, 'source', 'Entity'),
                        "target": getattr(st, 'target', 'Entity')
                    })
                    
                ontology_schema["edge_types"].append({
                    "name": edge_name,
                    "description": edge_class.__doc__ or '',
                    "source_targets": st_list,
                    "attributes": attributes
                })
                
        for graph_id in graph_ids:
            graph = self.client._load_graph(graph_id)
            graph["ontology"] = ontology_schema
            self.client._save_graph(graph_id, graph)
            logger.info(f"LocalGraph: Ontology set for graph '{graph_id}'. Schema: {ontology_schema}")
            
    def add_batch(self, graph_id: str, episodes: List[Any]) -> List[LocalEpisode]:
        graph = self.client._load_graph(graph_id)
        local_episodes = []
        
        for ep in episodes:
            ep_data = getattr(ep, 'data', None) or (ep.get('data') if isinstance(ep, dict) else str(ep))
            ep_type = getattr(ep, 'type', 'text') or (ep.get('type') if isinstance(ep, dict) else 'text')
            
            ep_uuid = f"ep_{uuid.uuid4().hex[:16]}"
            
            graph["episodes"][ep_uuid] = {
                "uuid": ep_uuid,
                "data": ep_data,
                "type": ep_type,
                "processed": True
            }
            
            # Extract and merge
            self.client._extract_and_merge(graph_id, graph, ep_data)
            
            local_episodes.append(LocalEpisode(ep_uuid, ep_data, ep_type))
            
        self.client._save_graph(graph_id, graph)
        return local_episodes
        
    def add(self, graph_id: str, type: str, data: str):
        self.add_batch(graph_id, [{"data": data, "type": type}])
        
    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges", reranker: str = None) -> LocalSearchResult:
        graph = self.client._load_graph(graph_id)
        
        nodes_matches = []
        edges_matches = []
        
        query_words = [w.lower().strip() for w in query.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query.lower() in text_lower:
                return 100
            score = 0
            for w in query_words:
                if w in text_lower:
                    score += 10
            return score

        for node_uuid, node in graph.get("nodes", {}).items():
            score = match_score(node.get("name", "")) + match_score(node.get("summary", ""))
            if score > 0:
                nodes_matches.append((score, LocalNode(
                    uuid_str=node["uuid"],
                    name=node["name"],
                    labels=node["labels"],
                    summary=node["summary"],
                    attributes=node["attributes"],
                    created_at=node.get("created_at")
                )))
                
        for edge_uuid, edge in graph.get("edges", {}).items():
            score = match_score(edge.get("fact", "")) + match_score(edge.get("name", ""))
            if score > 0:
                edges_matches.append((score, LocalEdge(
                    uuid_str=edge["uuid"],
                    name=edge["name"],
                    fact=edge["fact"],
                    source_node_uuid=edge["source_node_uuid"],
                    target_node_uuid=edge["target_node_uuid"],
                    attributes=edge["attributes"],
                    created_at=edge.get("created_at"),
                    valid_at=edge.get("valid_at"),
                    invalid_at=edge.get("invalid_at"),
                    expired_at=edge.get("expired_at")
                )))
                
        nodes_matches.sort(key=lambda x: x[0], reverse=True)
        edges_matches.sort(key=lambda x: x[0], reverse=True)
        
        matched_nodes = [n for score, n in nodes_matches[:limit]]
        matched_edges = [e for score, e in edges_matches[:limit]]
        
        if scope == "nodes":
            return LocalSearchResult(edges=[], nodes=matched_nodes)
        elif scope == "edges":
            return LocalSearchResult(edges=matched_edges, nodes=[])
        else:
            return LocalSearchResult(edges=matched_edges, nodes=matched_nodes)

class LocalZep:
    def __init__(self):
        self.graph = LocalGraphManager(self)
        self._llm = None
        self.graphs_dir = os.path.join(os.path.dirname(__file__), '../../uploads/graphs')
        os.makedirs(self.graphs_dir, exist_ok=True)
        
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm
        
    def _get_filepath(self, graph_id: str) -> str:
        return os.path.join(self.graphs_dir, f"{graph_id}.json")
        
    def _list_graph_ids(self) -> List[str]:
        if not os.path.exists(self.graphs_dir):
            return []
        return [f.split('.')[0] for f in os.listdir(self.graphs_dir) if f.endswith('.json')]
        
    def _load_graph(self, graph_id: str) -> Dict[str, Any]:
        filepath = self._get_filepath(graph_id)
        if not os.path.exists(filepath):
            return {
                "graph_id": graph_id,
                "name": "",
                "description": "",
                "ontology": {},
                "nodes": {},
                "edges": {},
                "episodes": {}
            }
        
        lock = _get_graph_lock(graph_id)
        with lock:
            for attempt in range(3):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    if attempt < 2:
                        logger.warning(f"JSON decode failed for {graph_id} (attempt {attempt+1}/3), retrying in 0.1s: {e}")
                        time.sleep(0.1)
                    else:
                        logger.error(f"Failed to load graph {graph_id} after multiple attempts due to JSON corruption: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error loading graph {graph_id}: {e}")
                    raise
            
    def _save_graph(self, graph_id: str, graph: Dict[str, Any]):
        filepath = self._get_filepath(graph_id)
        lock = _get_graph_lock(graph_id)
        with lock:
            temp_filepath = filepath + ".tmp"
            try:
                with open(temp_filepath, 'w', encoding='utf-8') as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)
                os.replace(temp_filepath, filepath)
            except Exception as e:
                logger.error(f"Failed to save graph {graph_id} atomically: {e}")
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except Exception:
                        pass
                raise

            
    def _extract_and_merge(self, graph_id: str, graph: Dict[str, Any], text: str):
        ontology = graph.get("ontology")
        if not ontology or (not ontology.get("entity_types") and not ontology.get("edge_types")):
            logger.warning(f"LocalGraph: No ontology configured for graph '{graph_id}', skipping LLM extraction.")
            return
            
        logger.info(f"LocalGraph: Extracting knowledge from text segment of length {len(text)}...")
        
        prompt = EXTRACT_PROMPT.format(
            ontology_schema=json.dumps(ontology, ensure_ascii=False, indent=2),
            text=text
        )
        
        messages = [
            {"role": "system", "content": "You are a precise Knowledge Graph extractor. Extract only what is present in the text according to the ontology schema. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            extraction_result = self.llm.chat_json(messages, temperature=0.1)
            
            node_name_to_uuid = {}
            for node_uuid, node in graph.get("nodes", {}).items():
                node_name_to_uuid[node["name"].lower().strip()] = node_uuid
                
            entities = extraction_result.get("entities", [])
            logger.info(f"LocalGraph: Extracted {len(entities)} entities.")
            
            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                type_name = ent.get("type", "Entity")
                summary = ent.get("summary", "")
                attributes = ent.get("attributes", {})
                
                name_key = name.lower()
                if name_key in node_name_to_uuid:
                    node_uuid = node_name_to_uuid[name_key]
                    existing_node = graph["nodes"][node_uuid]
                    
                    labels = list(set(existing_node.get("labels", []) + [type_name, "Entity", "Node"]))
                    existing_node["labels"] = labels
                    
                    if summary and summary not in existing_node.get("summary", ""):
                        existing_node["summary"] = (existing_node["summary"] + "; " + summary).strip("; ")
                        
                    for attr_k, attr_v in attributes.items():
                        if attr_v is not None:
                            existing_node["attributes"][attr_k] = attr_v
                else:
                    node_uuid = f"node_{uuid.uuid4().hex[:16]}"
                    graph["nodes"][node_uuid] = {
                        "uuid": node_uuid,
                        "name": name,
                        "labels": [type_name, "Entity", "Node"],
                        "summary": summary,
                        "attributes": attributes,
                        "created_at": datetime.now().isoformat()
                    }
                    node_name_to_uuid[name_key] = node_uuid
                    
            relationships = extraction_result.get("relationships", [])
            logger.info(f"LocalGraph: Extracted {len(relationships)} relationships.")
            
            for rel in relationships:
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                rel_type = rel.get("relationship", "").strip().upper()
                fact = rel.get("fact", "").strip()
                rel_attributes = rel.get("attributes", {})
                
                if not source_name or not target_name or not rel_type:
                    continue
                    
                source_uuid = node_name_to_uuid.get(source_name.lower())
                target_uuid = node_name_to_uuid.get(target_name.lower())
                
                if not source_uuid or not target_uuid:
                    logger.debug(f"LocalGraph: Skipping relationship '{rel_type}' because source '{source_name}' or target '{target_name}' was not resolved to an entity node.")
                    continue
                    
                edge_uuid = None
                for existing_uuid, edge in graph.get("edges", {}).items():
                    if edge.get("source_node_uuid") == source_uuid and edge.get("target_node_uuid") == target_uuid and edge.get("name") == rel_type:
                        edge_uuid = existing_uuid
                        break
                        
                if edge_uuid:
                    existing_edge = graph["edges"][edge_uuid]
                    if fact and fact not in existing_edge.get("fact", ""):
                        existing_edge["fact"] = (existing_edge["fact"] + "; " + fact).strip("; ")
                    for attr_k, attr_v in rel_attributes.items():
                        if attr_v is not None:
                            existing_edge["attributes"][attr_k] = attr_v
                else:
                    edge_uuid = f"edge_{uuid.uuid4().hex[:16]}"
                    graph["edges"][edge_uuid] = {
                        "uuid": edge_uuid,
                        "name": rel_type,
                        "fact": fact,
                        "source_node_uuid": source_uuid,
                        "target_node_uuid": target_uuid,
                        "attributes": rel_attributes,
                        "created_at": datetime.now().isoformat(),
                        "valid_at": datetime.now().isoformat(),
                        "invalid_at": None,
                        "expired_at": None
                    }
                    
        except Exception as e:
            logger.error(f"LocalGraph: Failed to extract knowledge graph: {e}", exc_info=True)

def get_zep_client(api_key: Optional[str] = None) -> Any:
    zep_key = api_key or Config.ZEP_API_KEY
    if not zep_key or zep_key.lower() == 'local':
        logger.info("Initializing LocalZep client (Fully Local, no Zep Cloud dependency).")
        return LocalZep()
    else:
        try:
            from zep_cloud.client import Zep
            logger.info("Initializing Zep Cloud client using ZEP_API_KEY.")
            return Zep(api_key=zep_key)
        except ImportError:
            logger.warning("zep_cloud library not found. Falling back to LocalZep client.")
            return LocalZep()
