from pymongo import MongoClient

client = MongoClient('mongodb://mongo:27017/')
db = client['syzbot_DB']

# Check attack join: baseline kcb_bug_name vs attack kshs_poclog_name
baseline_attack_names = set()
for doc in db['kernel_convert_baseline'].find(
        {'kcb_seq_lables': 'Attach'}, {'kcb_bug_name': 1}):
    baseline_attack_names.add(doc.get('kcb_bug_name', ''))

attack_strace_names = set()
for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find(
        {}, {'kshs_poclog_name': 1}):
    attack_strace_names.add(doc.get('kshs_poclog_name', ''))

overlap = baseline_attack_names & attack_strace_names
print(f"Baseline attack names:   {len(baseline_attack_names)}")
print(f"Attack strace poc names: {len(attack_strace_names)}")
print(f"Direct overlap:          {len(overlap)}")
print(f"\nSample baseline attack names: {list(baseline_attack_names)[:5]}")
print(f"Sample attack strace names:   {list(attack_strace_names)[:5]}")

# Check if attack strace has 'kshs_bugpoc_syscall_list' always populated
nonempty_list = db['kernel_syscallhook_bugpoc_trace_sum'].count_documents(
    {'kshs_bugpoc_syscall_list': {'$nin': ['', None]}})
print(f"\nAttack docs with non-empty syscall_list: {nonempty_list} / {len(attack_strace_names)}")

# Sample syscall_list for attack
print("\nSample attack syscall_list (name-based, first 3):")
for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find(
        {}, {'kshs_poclog_name': 1, 'kshs_bugpoc_syscall_list': 1}, limit=3):
    lst = str(doc.get('kshs_bugpoc_syscall_list', ''))[:200]
    print(f"  poc={doc.get('kshs_poclog_name')!r}")
    print(f"  list={lst!r}")
