from pymongo import MongoClient

client = MongoClient('mongodb://mongo:27017/')
db = client['syzbot_DB']

print("=== BASELINE: Normal entries (label=Normal) — first 5 ===")
for doc in db['kernel_convert_baseline'].find(
        {'kcb_seq_lables': 'Normal'},
        {'kcb_bug_name': 1, 'kcb_seq_lables': 1, 'kcb_master_line_ver': 1}, limit=5):
    print(f"  kcb_bug_name={doc.get('kcb_bug_name')!r}  label={doc.get('kcb_seq_lables')!r}")

print(f"\n=== BASELINE counts ===")
total = db['kernel_convert_baseline'].count_documents({})
normal = db['kernel_convert_baseline'].count_documents({'kcb_seq_lables': 'Normal'})
attack = db['kernel_convert_baseline'].count_documents({'kcb_seq_lables': 'Attach'})
print(f"  Total={total}  Normal={normal}  Attack={attack}")

print(f"\n=== NORMAL strace: count and file name samples ===")
n_count = db['kernel_syscall_normal_strace'].count_documents({})
print(f"  Total docs: {n_count}")

print(f"\n=== ATTACK strace: count and mlcode coverage ===")
a_count = db['kernel_syscallhook_bugpoc_trace_sum'].count_documents({})
nonempty_mlcode = db['kernel_syscallhook_bugpoc_trace_sum'].count_documents(
    {'kshs_bugpoc_syscall_mlcode': {'$nin': ['', None]}})
print(f"  Total docs: {a_count}")
print(f"  Non-empty mlcode: {nonempty_mlcode}")

# Sample attack docs with non-empty mlcode
print(f"\n=== ATTACK: docs WITH non-empty mlcode ===")
for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find(
        {'kshs_bugpoc_syscall_mlcode': {'$nin': ['', None]}},
        {'kshs_poclog_name': 1, 'kshs_bugpoc_syscall_mlcode': 1}, limit=3):
    ml = str(doc.get('kshs_bugpoc_syscall_mlcode', ''))[:150]
    print(f"  name={doc.get('kshs_poclog_name')!r}  mlcode={ml!r}")

# Check if baseline normal kcb_bug_name matches normal strace kns_normal_file_name
print(f"\n=== JOIN test: baseline Normal kcb_bug_name vs normal strace filenames ===")
baseline_normal_names = set()
for doc in db['kernel_convert_baseline'].find(
        {'kcb_seq_lables': 'Normal'}, {'kcb_bug_name': 1}):
    baseline_normal_names.add(doc.get('kcb_bug_name', ''))

normal_strace_names = set()
for doc in db['kernel_syscall_normal_strace'].find({}, {'kns_normal_file_name': 1}):
    normal_strace_names.add(doc.get('kns_normal_file_name', ''))

overlap = baseline_normal_names & normal_strace_names
print(f"  Baseline normal names: {len(baseline_normal_names)}")
print(f"  Normal strace filenames: {len(normal_strace_names)}")
print(f"  Direct overlap: {len(overlap)}")
print(f"  Sample baseline normal names: {list(baseline_normal_names)[:5]}")
print(f"  Sample normal strace names: {list(normal_strace_names)[:5]}")

# Try stripping extension from strace names
strace_noext = {n.rsplit('.', 1)[0]: n for n in normal_strace_names}
overlap2 = baseline_normal_names & set(strace_noext.keys())
print(f"  Overlap if baseline vs strace-no-ext: {len(overlap2)}")
# sample
for k in list(overlap2)[:3]:
    print(f"    baseline={k!r} → strace={strace_noext[k]!r}")
