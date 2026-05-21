from pymongo import MongoClient

client = MongoClient('mongodb://mongo:27017/')
db = client['syzbot_DB']

print("=== NORMAL collection (first 2 docs) ===")
for doc in db['kernel_syscall_normal_strace'].find({}, limit=2):
    del doc['_id']
    for k, v in doc.items():
        vtype = type(v).__name__
        vpreview = str(v)[:150]
        print(f"  {k}: [{vtype}] {vpreview}")
    print("  ---")

print("\n=== ATTACK collection (first 2 docs) ===")
for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find({}, limit=2):
    del doc['_id']
    for k, v in doc.items():
        vtype = type(v).__name__
        vpreview = str(v)[:150]
        print(f"  {k}: [{vtype}] {vpreview}")
    print("  ---")

print("\n=== BASELINE collection (first 2 docs) ===")
for doc in db['kernel_convert_baseline'].find({}, limit=2):
    del doc['_id']
    for k, v in doc.items():
        vtype = type(v).__name__
        vpreview = str(v)[:150]
        print(f"  {k}: [{vtype}] {vpreview}")
    print("  ---")

print("\n=== NORMAL: check kns_normal_mlseq_list on 5 docs ===")
for doc in db['kernel_syscall_normal_strace'].find(
        {}, {'kns_normal_file_name': 1, 'kns_normal_mlseq_list': 1}, limit=5):
    fname = doc.get('kns_normal_file_name', 'MISSING')
    mlseq = doc.get('kns_normal_mlseq_list', 'MISSING')
    mltype = type(mlseq).__name__
    mlpreview = str(mlseq)[:200]
    print(f"  fname={fname!r}  mlseq_type={mltype}  mlseq={mlpreview}")

print("\n=== ATTACK: check kshs_bugpoc_syscall_mlcode on 5 docs ===")
for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find(
        {}, {'kshs_poclog_name': 1, 'kshs_bugpoc_syscall_mlcode': 1}, limit=5):
    pname = doc.get('kshs_poclog_name', 'MISSING')
    mlcode = doc.get('kshs_bugpoc_syscall_mlcode', 'MISSING')
    mltype = type(mlcode).__name__
    mlpreview = str(mlcode)[:200]
    print(f"  pocname={pname!r}  mlcode_type={mltype}  mlcode={mlpreview}")

print("\n=== BASELINE: check kcb_bug_name on 5 docs ===")
for doc in db['kernel_convert_baseline'].find(
        {}, {'kcb_bug_name': 1, 'kcb_seq_lables': 1}, limit=5):
    bname = doc.get('kcb_bug_name', 'MISSING')
    label = doc.get('kcb_seq_lables', 'MISSING')
    print(f"  kcb_bug_name={bname!r}  label={label!r}")
