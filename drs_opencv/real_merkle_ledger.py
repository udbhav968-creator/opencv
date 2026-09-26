# real_merkle_ledger.py
"""
real_merkle_ledger.py
---------------------
GENUINE Cryptographic Merkle Tree Ledger for DRS Decision Audit Trails.

Implements:
  - Cryptographic SHA-256 Merkle Tree data structure
  - Leaf hash generation: H(Job_ID || Timestamp || Decision)
  - Recursive pairwise parent hash aggregation to Merkle Root
  - Merkle Audit Proof path generation (siblings in the tree)
  - Cryptographic Proof Verification: verify_merkle_proof(leaf, proof, root) -> True/False
"""

import hashlib
import time

class RealMerkleDRSLedger:
    """
    Genuine Cryptographic Merkle Tree Audit Ledger.
    Zero mock hashes — true cryptographic tree building and audit path verification.
    """

    def __init__(self):
        self.records = []
        self.leaves = []
        self.tree_levels = []
        self.merkle_root = None

    def _hash(self, data_str):
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def add_decision(self, job_id, final_call, timestamp=None):
        """Adds a genuine match decision to the cryptographic ledger."""
        ts = timestamp or time.time()
        record = {
            "job_id": job_id,
            "final_call": final_call,
            "timestamp": ts
        }
        self.records.append(record)
        leaf_hash = self._hash(f"{job_id}:{final_call}:{ts}")
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return leaf_hash

    def _rebuild_tree(self):
        """Constructs full Merkle Tree layers up to root."""
        if not self.leaves:
            self.merkle_root = None
            return

        current_level = self.leaves[:]
        self.tree_levels = [current_level]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i+1] if (i+1) < len(current_level) else left
                parent = self._hash(left + right)
                next_level.append(parent)
            current_level = next_level
            self.tree_levels.append(current_level)

        self.merkle_root = current_level[0]

    def get_audit_proof(self, leaf_index):
        """
        Generates a genuine cryptographic audit proof path for a leaf.
        """
        if leaf_index >= len(self.leaves):
            return None

        proof = []
        idx = leaf_index
        for level in self.tree_levels[:-1]:
            is_right = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right else (idx + 1 if (idx + 1) < len(level) else idx)
            proof.append({
                "sibling_hash": level[sibling_idx],
                "direction": "left" if is_right else "right"
            })
            idx = idx // 2

        return {
            "leaf_hash": self.leaves[leaf_index],
            "merkle_root": self.merkle_root,
            "proof_path": proof,
            "verified": self.verify_proof(self.leaves[leaf_index], proof, self.merkle_root)
        }

    @staticmethod
    def verify_proof(leaf_hash, proof_path, expected_root):
        """
        Mathematically verifies a Merkle proof path against the root hash.
        """
        current = leaf_hash
        for step in proof_path:
            sib = step["sibling_hash"]
            if step["direction"] == "left":
                current = hashlib.sha256((sib + current).encode('utf-8')).hexdigest()
            else:
                current = hashlib.sha256((current + sib).encode('utf-8')).hexdigest()

        return current == expected_root
