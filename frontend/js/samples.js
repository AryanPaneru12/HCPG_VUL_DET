/**
 * HCPG-GNN Auditor v4.0 — Sample Smart Contracts
 * Pre-built vulnerability examples for quick testing
 */

const SAMPLES = {
  reentrancy: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// VULNERABLE: Cross-Function Reentrancy (SWC-107)
contract ReentrancyVault {
    mapping(address => uint256) private balances;
    bool private locked;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint256 bal = balances[msg.sender];
        require(bal > 0, "No balance");
        (bool ok,) = msg.sender.call{value: bal}("");
        require(ok);
        balances[msg.sender] = 0;
    }

    function emergencyWithdraw(address payable to) external {
        withdraw();
        to.transfer(address(this).balance);
    }
}`,

  access: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// VULNERABLE: Access Control (SWC-115)
contract TokenSale {
    address public admin;
    mapping(address => uint256) public tokens;
    uint256 public price = 1 ether;

    constructor() { admin = msg.sender; }

    function buy() external payable {
        tokens[msg.sender] += msg.value / price;
    }

    function setPrice(uint256 newPrice) external {
        price = newPrice;
    }

    function withdraw() external {
        payable(msg.sender).transfer(address(this).balance);
    }

    function adminAction() external {
        require(msg.sender == admin);
        setPrice(0);
    }
}`,

  tod: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// VULNERABLE: Transaction Order Dependency (SWC-114)
contract RaceAuction {
    address public highestBidder;
    uint256 public highestBid;
    mapping(address => uint256) public bids;

    function bid() external payable {
        require(msg.value > highestBid);
        if (highestBidder != address(0)) {
            payable(highestBidder).transfer(highestBid);
        }
        highestBidder = msg.sender;
        highestBid = msg.value;
        bids[msg.sender] = msg.value;
    }

    function claimReward() external {
        require(msg.sender == highestBidder);
        bids[msg.sender] = 0;
    }
}`,

  clean: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// SAFE: Best practices implemented
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract SafeBank is ReentrancyGuard, Ownable {
    mapping(address => uint256) private balances;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    function deposit() external payable {
        require(msg.value > 0, "Zero deposit");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount, "Insufficient");
        balances[msg.sender] -= amount;
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        emit Withdrawal(msg.sender, amount);
    }

    function adminWithdraw(uint256 amount) external onlyOwner {
        payable(owner()).transfer(amount);
    }
}`
};

/**
 * Load a sample contract into the editor
 * @param {string} key - Sample type key
 */
function loadSample(key) {
  const area = document.getElementById('codeArea');
  area.value = SAMPLES[key];
  const lines = SAMPLES[key].split('\n').length;
  document.getElementById('lineCount').textContent = lines + ' lines';
  clearResults();
}
