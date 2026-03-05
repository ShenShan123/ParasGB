import os
import glob
import torch
import re
import numpy as np
import traceback
from torch_geometric.data import HeteroData

# ================= 配置区域 =================
# 输入数据的根目录
INPUT_ROOT = "D:/desktop/github_push/rcg_v2/RCcases" 
# 输出数据的保存目录
OUTPUT_DIR = "data"

# --- Pin-Pin 连边策略配置 ---
# 阈值：如果一个 Net 的 Pin 数量少于此值，则进行全连接（100% 物理真值）
SMALL_NET_THRESHOLD = 64 
# 对于超过阈值的大网，每个 Pin 最多保留多少个电阻最小的邻居
MAX_NEIGHBORS_LARGE = 10
# 物理截断：忽略大于 1M Ohm 的微弱连接（视为断路）
RESISTANCE_CUTOFF = 1e6 
# ===========================================

# ================= 1. 基础解析工具 =================

def normalize_name(name):
    if not name: return ""
    return name.upper().replace('\\', '').replace('%', '').strip()

def parse_val(val_str):
    if not val_str: return 0.0
    val_str = val_str.lower().strip()
    mults = {'t':1e12, 'g':1e9, 'meg':1e6, 'x':1e6, 'k':1e3, 'm':1e-3, 'u':1e-6, 'n':1e-9, 'p':1e-12, 'f':1e-15, 'a':1e-18}
    try: return float(val_str)
    except: pass
    match = re.match(r'^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-z]*)$', val_str)
    if not match: return 0.0
    num, suffix = match.groups()
    try: val = float(num)
    except: return 0.0
    if suffix:
        if suffix.startswith('meg'): return val * 1e6
        if suffix[0] in mults: return val * mults[suffix[0]]
    return val

def read_file_tokens(filepath):
    if not os.path.exists(filepath): return []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.readlines()
    lines = []
    buffer = ""
    for l in raw:
        s = l.strip()
        if not s or s.startswith(('*','//','$')): continue
        if s.startswith('+'): buffer += " " + s[1:].strip()
        elif buffer.endswith('\\'): buffer = buffer[:-1].strip() + " " + s
        else:
            if buffer: lines.append(buffer)
            buffer = s
    if buffer: lines.append(buffer)
    tokenized = []
    for l in lines:
        clean = l.replace('(', ' ').replace(')', ' ')
        toks = clean.split()
        if toks: tokenized.append(toks)
    return tokenized

def solve_res_numpy(res_list, ports):
    """
    使用矩阵运算计算端口间的等效电阻（考虑并联效应）
    返回: (src_indices, dst_indices, resistance_values)
    """
    if not res_list or len(ports) < 2: return [], [], []
    nodes = set()
    for n1, n2, _ in res_list: nodes.add(n1); nodes.add(n2)
    node_map = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    port_indices = []
    for i, p in enumerate(ports):
        if p in node_map: port_indices.append((node_map[p], i))
    
    if len(port_indices) < 2: return [], [], []

    # 构建电导矩阵 G
    G = np.zeros((N, N))
    for n1, n2, r in res_list:
        r = max(r, 1e-3) # 防止除零
        g = 1.0 / r
        u, v = node_map[n1], node_map[n2]
        G[u, u] += g; G[v, v] += g; G[u, v] -= g; G[v, u] -= g
        
    try:
        # 计算阻抗矩阵 Z
        G_red = G[:-1, :-1]
        Z_red = np.linalg.inv(G_red)
        ref = N-1
        
        def get_z(i, j): 
            return 0.0 if (i==ref or j==ref) else Z_red[i, j]
            
        srcs, dsts, attrs = [], [], []
        # 计算任意两个端口间的等效电阻
        for i1 in range(len(port_indices)):
            for i2 in range(i1+1, len(port_indices)):
                um, up = port_indices[i1] # um: matrix idx, up: port idx
                vm, vp = port_indices[i2]
                req = get_z(um, um) + get_z(vm, vm) - 2*get_z(um, vm)
                srcs.append(up); dsts.append(vp); attrs.append(req)
        return srcs, dsts, attrs
    except: 
        return [], [], []

# ================= 2. 核心处理逻辑 =================

def process_single_case(sp_path, pxi_path, pex_path, case_id):
    pex_lines = read_file_tokens(pex_path)
    pxi_lines = read_file_tokens(pxi_path)
    sp_lines  = read_file_tokens(sp_path)

    # --- PEX 解析 (提取子电路与寄生参数) ---
    subckt_info = {}
    curr = None
    for toks in pex_lines:
        head = toks[0].upper()
        if head in ['SUBCKT', '.SUBCKT']:
            if len(toks) < 2: continue
            curr = normalize_name(toks[1])
            ports = [normalize_name(t) for t in toks[2:] if '=' not in t and t != '\\']
            subckt_info[curr] = {'ports': ports, 'cap': 0.0, 'res': []}
        elif head in ['ENDS', '.ENDS']: curr = None
        elif curr:
            if head.startswith('C'):
                val = 0.0
                for t in toks: 
                    if t.lower().startswith('c='): val = parse_val(t.split('=')[1]); break
                if val == 0: 
                    for t in toks[1:]:
                        if t.lower() != 'capacitor' and normalize_name(t) not in subckt_info[curr]['ports']:
                            v = parse_val(t); 
                            if v > 0: val = v; break
                if val > 0: subckt_info[curr]['cap'] += val
            elif head.startswith('R'):
                r_val = 0.0
                for t in toks:
                    if t.lower().startswith('r='): r_val = parse_val(t.split('=')[1]); break
                if r_val == 0:
                    for t in toks[3:]:
                        if t.lower() != 'resistor': 
                            v = parse_val(t); 
                            if v > 0: r_val = v; break
                if len(toks) >= 3:
                    subckt_info[curr]['res'].append((normalize_name(toks[1]), normalize_name(toks[2]), r_val))

    # --- PXI 解析 (映射) ---
    sp_map = {}
    for toks in pxi_lines:
        if toks[0].lower().startswith('x_'):
            sub = normalize_name(toks[-1])
            nets = [normalize_name(t) for t in toks[1:-1]]
            if sub in subckt_info:
                for i in range(min(len(nets), len(subckt_info[sub]['ports']))):
                    sp_map[nets[i]] = (sub, i)

    # --- SP 解析 (器件提取) ---
    dev_data, dev_conns, dev_types = [], [], []
    
    def get_p(tokens, k): 
        for t in tokens: 
            if t.lower().startswith(k+'='): return parse_val(t.split('=')[1])
        return 0

    for toks in sp_lines:
        name = normalize_name(toks[0])
        p = {'w':0,'l':0,'m':1,'nf':1,'is_p':0,'type':''}
        for k in ['ad','as','pd','ps','nrd','nrs','sa','sb']: p[k] = get_p(toks, k)
        p['w']=get_p(toks, 'w'); p['l']=get_p(toks, 'l')
        p['m']=get_p(toks, 'm') or 1; p['nf']=get_p(toks, 'nf') or 1
        
        nets = []
        if name.startswith('M'):
            p['type']='mos'
            if len(toks)>=6:
                nets = [normalize_name(x) for x in toks[1:5]]
                if 'p' in toks[5].lower(): p['is_p']=1
        elif name.startswith('R'):
            p['type']='res'
            if len(toks)>=3: nets = [normalize_name(x) for x in toks[1:3]]
            if get_p(toks, 'r') == 0:
                for t in toks[3:]:
                    if '=' in t: continue
                    val = parse_val(t)
                    if val > 0: break
        elif name.startswith('C') or name.startswith('MIM'):
            p['type']='cap'
            if len(toks)>=3: nets = [normalize_name(x) for x in toks[1:3]]
        else: continue
        
        feat = [
            1 if p['type']=='mos' else 0, 1 if p['type']=='res' else 0, 1 if p['type']=='cap' else 0,
            p['w'], p['l'], p['m'], p['nf'], 
            p.get('ad',0), p.get('as',0), p.get('pd',0), p.get('ps',0), 
            p.get('nrd',0), p.get('nrs',0), p.get('sa',0), p.get('sb',0),
            p['is_p']
        ]
        dev_data.append(feat)
        dev_conns.append(nets); dev_types.append(p['type'])

    # --- 构建图结构 ---
    net_names = list(subckt_info.keys())
    net_map_idx = {n:i for i,n in enumerate(net_names)}
    pin_list, pin_lookup = [], {}
    for n in net_names:
        for port in subckt_info[n]['ports']:
            pid = len(pin_list)
            pin_list.append((net_map_idx[n], port))
            pin_lookup[(n, port)] = pid

    if len(dev_data) == 0: dev_feats = torch.empty((0, 16), dtype=torch.float)
    else: dev_feats = torch.tensor(dev_data, dtype=torch.float)
         
    net_feats = torch.zeros((len(net_names), 10))
    net_labels = torch.zeros((len(net_names), 1))
    pin_feats = torch.zeros((len(pin_list), 6))

    for i, n in enumerate(net_names):
        net_labels[i] = subckt_info[n]['cap']
        if 'VDD' in n or 'PWR' in n: net_feats[i,0]=1
        elif 'VSS' in n or 'GND' in n: net_feats[i,1]=1
        else: net_feats[i,2]=1
        if subckt_info[n]['ports']: net_feats[i,3]=1

    edge_dp_s, edge_dp_d = [], []
    edge_dn_s, edge_dn_d = [], []
    
    for di, nets in enumerate(dev_conns):
        dt = dev_types[di]
        w, l = dev_data[di][3], dev_data[di][4]
        conn_nets = set()
        for ti, net in enumerate(nets):
            if net in sp_map:
                sub, pidx = sp_map[net]
                ports = subckt_info[sub]['ports']
                if pidx < len(ports):
                    pname = ports[pidx]
                    if (sub, pname) in pin_lookup:
                        pid = pin_lookup[(sub, pname)]
                        nid = net_map_idx[sub]
                        edge_dp_s.append(di); edge_dp_d.append(pid)
                        conn_nets.add(nid)
                        if dt=='mos' and ti<4: pin_feats[pid, ti]=1
                        elif dt=='res': pin_feats[pid, 5]=1
                        elif dt=='cap': pin_feats[pid, 4]=1
                        if dt=='mos':
                            if ti==1: net_feats[nid, 5]+=1
                            elif ti in [0,2]: net_feats[nid, 6]+=1
                            elif ti==3: net_feats[nid, 7]+=1
                            net_feats[nid, 8]+=l; net_feats[nid, 9]+=w
        if dt=='mos':
            for nid in conn_nets: net_feats[nid, 4]+=1
        for nid in conn_nets: edge_dn_s.append(di); edge_dn_d.append(nid)

    edge_pn_s = list(range(len(pin_list)))
    edge_pn_d = [p[0] for p in pin_list]

    # === 关键修正：自适应 Pin-Pin 连边 ===
    edge_pp_s, edge_pp_d, edge_pp_a = [], [], []

    for n in net_names:
        ports = subckt_info[n]['ports']
        num_ports = len(ports)
        
        # 1. 计算物理真值 (Matrix Inversion)
        srcs, dsts, reqs = solve_res_numpy(subckt_info[n]['res'], ports)
        if not srcs: continue

        base = pin_lookup[(n, ports[0])]
        
        # 2. 根据网的大小选择策略
        if num_ports <= SMALL_NET_THRESHOLD:
            # --- 策略 A: 小网全量 (Full Clique) ---
            # 物理上精确，保留所有计算出的边
            for s, d, r in zip(srcs, dsts, reqs):
                if r > RESISTANCE_CUTOFF: continue
                # 双向添加
                edge_pp_s.extend([base+s, base+d])
                edge_pp_d.extend([base+d, base+s])
                edge_pp_a.extend([r, r])
        else:
            # --- 策略 B: 大网优化 (Top-K) ---
            # 避免显存爆炸，只保留最强的 K 个物理连接
            adj = {i: [] for i in range(num_ports)}
            for s, d, r in zip(srcs, dsts, reqs):
                if r > RESISTANCE_CUTOFF: continue
                adj[s].append((d, r))
                adj[d].append((s, r))
            
            processed_pairs = set()
            for u in range(num_ports):
                # 排序并取 Top-K
                neighbors = sorted(adj[u], key=lambda x: x[1])[:MAX_NEIGHBORS_LARGE]
                for v, r in neighbors:
                    u_g, v_g = base + u, base + v
                    pair = tuple(sorted((u_g, v_g)))
                    if pair not in processed_pairs:
                        edge_pp_s.extend([u_g, v_g])
                        edge_pp_d.extend([v_g, u_g])
                        edge_pp_a.extend([r, r])
                        processed_pairs.add(pair)

    # === 构建 HeteroData ===
    data = HeteroData()
    data['dev'].x = dev_feats
    data['pin'].x = pin_feats
    data['net'].x = net_feats; data['net'].y = net_labels

    
    def to_long(l): return torch.tensor(l, dtype=torch.long) if l else torch.empty((0,), dtype=torch.long)
    def to_edge(s, d): 
        if not s: return torch.empty((2, 0), dtype=torch.long)
        return torch.stack([to_long(s), to_long(d)], dim=0)

    data['dev','connects_to','pin'].edge_index = to_edge(edge_dp_s, edge_dp_d)    
    data['dev','connects_to','net'].edge_index = to_edge(edge_dn_s, edge_dn_d)
    data['pin','belongs_to','net'].edge_index = to_edge(edge_pn_s, edge_pn_d)
    
    # 构建 Pin-Pin 边
    if edge_pp_s:
        data['pin','pair_to','pin'].edge_index = to_edge(edge_pp_s, edge_pp_d)
        data['pin','pair_to','pin'].y = torch.tensor(edge_pp_a, dtype=torch.float).view(-1, 1)
    else:
        data['pin','pair_to','pin'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['pin','pair_to','pin'].y = torch.empty((0, 1), dtype=torch.float)

    data.case_id = case_id
    return data

# ================= 3. 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(INPUT_ROOT):
        print(f"错误: 找不到输入目录 '{INPUT_ROOT}'"); return

    subdirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d)) and d.isdigit()]
    subdirs.sort(key=lambda x: int(x))
    print(f"检测到 Case: {subdirs}\n")

    for folder_name in subdirs:
        case_path = os.path.join(INPUT_ROOT, folder_name)
        sp_files = glob.glob(os.path.join(case_path, "*.sp"))
        pxi_files = glob.glob(os.path.join(case_path, "*.pxi"))
        pex_files = glob.glob(os.path.join(case_path, "*.pex"))

        if not (sp_files and pxi_files and pex_files):
            print(f"[跳过] Case {folder_name}: 文件缺失"); continue
            
        print(f"处理 Case {folder_name} ...")
        try:
            graph_data = process_single_case(sp_files[0], pxi_files[0], pex_files[0], folder_name)
            
            # 保存 .pt
            pt_path = os.path.join(OUTPUT_DIR, f"case{folder_name}_RC.pt")
            torch.save(graph_data, pt_path)
            
            # 保存 .txt 报告
            with open(os.path.join(OUTPUT_DIR, f"case{folder_name}_report.txt"), 'w') as f:
                f.write(str(graph_data))
            
            print(f"  -> 完成: {pt_path}")
        except Exception as e:
            print(f"  -> 失败: {e}"); traceback.print_exc()

    print("\n所有任务完成！")

if __name__ == '__main__':
    main()