import pandas as pd
import numpy as np
from neo4j import GraphDatabase

# ==============================================
# 1. 配置项（请修改为你自己的Neo4j账号密码）
# ==============================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4jforlearn"
EXCEL_PATH = "tunnel_health_gnn_binary_structure.xlsx"

# ==============================================
# 2. 初始化Neo4j驱动
# ==============================================
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 验证连接
try:
    driver.verify_connectivity()
    print("Neo4j连接成功")
except Exception as e:
    print(f"Neo4j连接失败，请检查服务状态与账号密码：{e}")
    exit()


# ==============================================
# 3. 导入前清理旧数据（重复运行不会产生重复数据，仅在第一次执行，后续更新千万不能做）
# 如果不在意规模，可不用做
# ==============================================
# def clear_old_data(tx):
#     tx.run("MATCH (n) DETACH DELETE n")

# with driver.session() as session:
#     session.execute_write(clear_old_data)
# print("旧数据清理完成")


# ==============================================
# 4. 读取Excel数据，处理空值适配Neo4j存储格式
# ==============================================
def clean_value(v):
    """将pandas的NaN转为Neo4j支持的空值None"""
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# 读取所有Sheet
defect_df = pd.read_excel(EXCEL_PATH, sheet_name="Defect_Node")
structure_df = pd.read_excel(EXCEL_PATH, sheet_name="Structure_Node")
edge_df = pd.read_excel(EXCEL_PATH, sheet_name="Edge_Relation")

# 转为字典列表，处理空值
# 转成字典列表做批量导入
defect_records = [{k: clean_value(v) for k,v in row.items()} for row in defect_df.to_dict("records")]
structure_records = [{k: clean_value(v) for k,v in row.items()} for row in structure_df.to_dict("records")]


# ==============================================
# 5. 批量导入节点
# ==============================================
def import_defect_nodes(tx, records):
    cypher = """
    UNWIND $records AS row
    MERGE (d:Defect {defect_id: row.defect_id})
    SET d = row
    """
    tx.run(cypher, records=records)

# 导入结构节点（不管区段/区间/线路，标签统一为Structure，通过granularity字段区分粒度）
def import_structure_nodes(tx, records):
    cypher = """
    UNWIND $records AS row
    MERGE (s:Structure {node_id: row.node_id})
    SET s = row
    """
    tx.run(cypher, records=records)


with driver.session() as session:
    session.execute_write(import_defect_nodes, defect_records)
    print(f"导入Defect节点：{len(defect_records)}个")
    session.execute_write(import_structure_nodes, structure_records)
    print(f"导入Structure节点：{len(structure_records)}个")


# ==============================================
# 6. 批量导入隶属关系（带边权重属性）
# ==============================================
def import_edges(tx, edge_df):
    # 三类边，后续也可根据需要修改
    edge_configs = [
        {
            "rel_name": "BELONGS_TO",       # 边类型名
            "src_label": "Defect",           # 源节点标签
            "src_id_key": "defect_id",       # 源节点唯一ID字段
            "tgt_label": "Structure",        # 目标节点标签
            "tgt_id_key": "node_id",         # 目标节点唯一ID字段
            "match_rule": lambda row: row["src_type"] == "defect" and row["tgt_type"] == "structure"
        },
        {
            "rel_name": "HAS",
            "src_label": "Structure",
            "src_id_key": "node_id",
            "tgt_label": "Defect",
            "tgt_id_key": "defect_id",
            "match_rule": lambda row: row["src_type"] == "structure" and row["tgt_type"] == "defect"
        },
        {
            "rel_name": "CONNECTS_TO",
            "src_label": "Structure",
            "src_id_key": "node_id",
            "tgt_label": "Structure",
            "tgt_id_key": "node_id",
            "match_rule": lambda row: row["src_type"] == "structure" and row["tgt_type"] == "structure"
        }
    ]

    # 逐类边批量导入
    for cfg in edge_configs:
        # 筛选当前类型的边
        subset = edge_df[edge_df.apply(cfg["match_rule"], axis=1)]
        if len(subset) == 0:
            continue
        # 整理导入记录
        records = [
            {"src_id": row["src_id"], "tgt_id": row["tgt_id"], "edge_weight": row["edge_weight"]}
            for _, row in subset.iterrows()
        ]
        # 写对应Cypher，直接按节点标签和唯一ID匹配，无模糊查询
        cypher = f"""
        UNWIND $records AS rec
        MATCH (a:{cfg["src_label"]} {{{cfg["src_id_key"]}: rec.src_id}}),
              (b:{cfg["tgt_label"]} {{{cfg["tgt_id_key"]}: rec.tgt_id}})
        MERGE (a)-[r:{cfg["rel_name"]}]->(b)
        SET r.edge_weight = rec.edge_weight
        """
        tx.run(cypher, records=records)
        print(f"  - 导入{cfg['rel_name']}关系：{len(records)}条")


with driver.session() as session:
    session.execute_write(import_edges, edge_df)
    print(f"导入边关系：{len(edge_df)}条")


# ==============================================
# 7. 创建唯一索引，加速后续GNN数据查询
# ==============================================
def create_indexes(tx):
    tx.run("CREATE INDEX defect_id_idx IF NOT EXISTS FOR (n:Defect) ON (n.defect_id)")
    tx.run("CREATE INDEX structure_id_idx IF NOT EXISTS FOR (n:Structure) ON (n.node_id)")


with driver.session() as session:
    session.execute_write(create_indexes)
    print("索引创建完成")

# 关闭连接
driver.close()
print("\n===== 数据导入全部完成 =====")
