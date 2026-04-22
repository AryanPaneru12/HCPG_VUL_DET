"""
Unit Tests for HCPG-GNN Backend API
Run:  python -m pytest tests/test_backend.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.app import app


client = TestClient(app)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

class TestHealthEndpoints:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "4.0.0"
        assert "HGT" in data["model_name"]

    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "HCPG-GNN" in data["message"]

    def test_model_info(self):
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "architecture" in data
        assert "training_metrics" in data
        assert data["architecture"]["num_classes"] == 5

    def test_vulnerability_list(self):
        resp = client.get("/api/vulnerabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "supported" in data
        assert len(data["supported"]) >= 5

    def test_benchmark(self):
        resp = client.get("/api/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        models = data["models"]
        hgt = next(m for m in models if "HGT" in m["name"])
        assert hgt["accuracy"] >= 95.0
        assert hgt["f1"] >= 0.90  # F1 ≥ 0.90, accuracy is the primary ≥95% target


# ============================================================================
# Sample Contracts
# ============================================================================

class TestSampleEndpoints:
    @pytest.mark.parametrize("sample_type", ["reentrancy", "access", "tod", "safe"])
    def test_get_sample(self, sample_type):
        resp = client.get(f"/api/samples/{sample_type}")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "code" in data
        assert len(data["code"]) > 50

    def test_invalid_sample(self):
        resp = client.get("/api/samples/nonexistent")
        assert resp.status_code == 404


# ============================================================================
# Contract Analysis
# ============================================================================

REENTRANCY_CONTRACT = """
pragma solidity ^0.6.0;
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    constructor() public { owner = msg.sender; }
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] -= amount;
    }
    function drainFunds(address payable target) public {
        target.transfer(address(this).balance);
    }
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
"""

SAFE_CONTRACT = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
contract SafeBank is ReentrancyGuard, Ownable {
    mapping(address => uint256) private balances;
    function deposit() external payable {
        require(msg.value > 0);
        balances[msg.sender] += msg.value;
    }
    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
    }
    function adminWithdraw(uint256 amount) external onlyOwner {
        payable(owner()).transfer(amount);
    }
}
"""


class TestContractAnalysis:
    def test_analyze_reentrancy(self):
        resp = client.post("/api/analyze", json={"source_code": REENTRANCY_CONTRACT})
        assert resp.status_code == 200
        data = resp.json()
        assert data["contract_name"] == "VulnerableBank"
        assert data["safe"] is False
        assert len(data["vulnerabilities"]) > 0
        # Should detect reentrancy
        vuln_types = [v["vulnerability_type"] for v in data["vulnerabilities"]]
        assert "Reentrancy" in vuln_types
        # Metrics should be present
        assert data["metrics"]["accuracy"] >= 0.90
        assert "graph_stats" in data

    def test_analyze_safe_contract(self):
        resp = client.post("/api/analyze", json={"source_code": SAFE_CONTRACT})
        assert resp.status_code == 200
        data = resp.json()
        assert data["contract_name"] == "SafeBank"
        # Safe contract should have fewer or no vulnerabilities
        assert data["metrics"]["accuracy"] >= 0.90

    def test_analyze_access_control(self):
        code = """
pragma solidity ^0.8.0;
contract TokenSale {
    address public admin;
    uint256 public price = 1 ether;
    constructor() { admin = msg.sender; }
    function setPrice(uint256 newPrice) external { price = newPrice; }
    function withdraw() external { payable(msg.sender).transfer(address(this).balance); }
}
"""
        resp = client.post("/api/analyze", json={"source_code": code})
        assert resp.status_code == 200
        data = resp.json()
        assert data["safe"] is False

    def test_analyze_tod(self):
        code = """
pragma solidity ^0.8.0;
contract RaceAuction {
    address public highestBidder;
    uint256 public highestBid;
    mapping(address => uint256) public bids;
    function bid() external payable {
        require(msg.value > highestBid);
        highestBidder = msg.sender;
        highestBid = msg.value;
        bids[msg.sender] = msg.value;
    }
    function claimReward() external {
        require(msg.sender == highestBidder);
        bids[msg.sender] = 0;
    }
}
"""
        resp = client.post("/api/analyze", json={"source_code": code})
        assert resp.status_code == 200
        data = resp.json()
        vuln_types = [v["vulnerability_type"] for v in data["vulnerabilities"]]
        assert "Transaction Order Dependency" in vuln_types

    def test_analyze_compat_endpoint(self):
        resp = client.post("/analyze", json={"source_code": REENTRANCY_CONTRACT})
        assert resp.status_code == 200

    def test_analyze_empty_source(self):
        resp = client.post("/api/analyze", json={"source_code": ""})
        assert resp.status_code == 400

    def test_analyze_short_source(self):
        resp = client.post("/api/analyze", json={"source_code": "hello"})
        assert resp.status_code == 400

    def test_response_structure(self):
        resp = client.post("/api/analyze", json={"source_code": REENTRANCY_CONTRACT})
        data = resp.json()
        # Verify all expected fields
        assert "contract_hash" in data
        assert "contract_name" in data
        assert "vulnerabilities" in data
        assert "metrics" in data
        assert "graph_stats" in data
        assert "analysis_time" in data
        assert "safe" in data
        assert "summary" in data
        assert "call_graph" in data
        assert "cfg_graph" in data
        # Verify graph stats
        gs = data["graph_stats"]
        assert gs["function_nodes"] > 0
        assert gs["total_nodes"] > 0

    def test_call_graph_structure(self):
        resp = client.post("/api/analyze", json={"source_code": REENTRANCY_CONTRACT})
        data = resp.json()
        cg = data["call_graph"]
        assert "nodes" in cg
        assert "edges" in cg
        assert len(cg["nodes"]) > 0

    def test_vulnerability_fields(self):
        resp = client.post("/api/analyze", json={"source_code": REENTRANCY_CONTRACT})
        data = resp.json()
        for vuln in data["vulnerabilities"]:
            assert "swc_id" in vuln
            assert "vulnerability_type" in vuln
            assert "severity" in vuln
            assert "confidence" in vuln
            assert "function_affected" in vuln
            assert "description" in vuln
            assert "remediation" in vuln
            assert 0.0 <= vuln["confidence"] <= 1.0
            assert vuln["severity"] in ["critical", "high", "medium", "low"]
