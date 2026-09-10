"""
EVMBypecode Preprocessing Module for Smart Contract Vulnerability Detection.

Supports both baseline (with opcode operand values) and ablation (without opcode operand values).
"""

# EVM Opcode Table
EVM_OPCODES = {}

# Populate PUSH1 - PUSH32 (0x60 - 0x7F)
for i in range(0x60, 0x80):
    n_bytes = i - 0x60 + 1
    EVM_OPCODES[i] = (f"PUSH{n_bytes}", n_bytes)

# Populate DUP1 - DUP16 (0x80 - 0x8F)
for i in range(0x80, 0x90):
    EVM_OPCODES[i] = (f"DUP{i - 0x80 + 1}", 0)

# Populate SWAP1 - SWAP16 (0x90 - 0x9F)
for i in range(0x90, 0xA0):
    EVM_OPCODES[i] = (f"SWAP{i - 0x90 + 1}", 0)

# Populate LOG0 - LOG4 (0xA0 - 0xA4)
for i in range(0xA0, 0xA5):
    EVM_OPCODES[i] = (f"LOG{i - 0xA0}", 0)

# Common Non-PUSH Opcodes
KNOWN_OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV", 0x05: "SDIV",
    0x06: "MOD", 0x07: "SMOD", 0x08: "ADDMOD", 0x09: "MULMOD", 0x0A: "EXP", 0x0B: "SIGNEXTEND",
    0x10: "LT", 0x11: "GT", 0x12: "SLT", 0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO",
    0x16: "AND", 0x17: "OR", 0x18: "XOR", 0x19: "NOT", 0x1A: "BYTE", 0x1B: "SHL", 0x1C: "SHR", 0x1D: "SAR",
    0x20: "SHA3", 0x30: "ADDRESS", 0x31: "BALANCE", 0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE",
    0x35: "CALLDATALOAD", 0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY", 0x38: "CODESIZE", 0x39: "CODECOPY",
    0x3A: "GASPRICE", 0x3B: "EXTCODESIZE", 0x3C: "EXTCODECOPY", 0x3D: "RETURNDATASIZE", 0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH", 0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER",
    0x44: "DIFFICULTY", 0x45: "GASLIMIT", 0x46: "CHAINID", 0x47: "SELFBALANCE", 0x48: "BASEFEE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8", 0x54: "SLOAD", 0x55: "SSTORE",
    0x56: "JUMP", 0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE", 0x5A: "GAS", 0x5B: "JUMPDEST",
    0x5C: "TLOAD", 0x5D: "TSTORE", 0x5E: "MCOPY", 0x5F: "PUSH0",
    0xF0: "CREATE", 0xF1: "CALL", 0xF2: "CALLCODE", 0xF3: "RETURN", 0xF4: "DELEGATECALL",
    0xF5: "CREATE2", 0xFA: "STATICCALL", 0xFD: "REVERT", 0xFE: "INVALID", 0xFF: "SELFDESTRUCT"
}

for op_val, name in KNOWN_OPCODES.items():
    EVM_OPCODES[op_val] = (name, 0)


def preprocess_bytecode(bytecode: str, use_opcode_values: bool = True, use_mnemonics: bool = False) -> str:
    """
    Preprocess raw EVM bytecode hex string.

    Args:
        bytecode: Raw hex string of compiled Solidity bytecode.
        use_opcode_values: If True, retain operand values. If False, strip operand values.
        use_mnemonics: If True, output EVM opcode names (e.g. PUSH1). If False, output hex tokens (e.g. 60).

    Returns:
        Space-separated token string for vectorization.
    """
    import re
    bytecode_str = str(bytecode).strip()
    if bytecode_str.startswith('0x') or bytecode_str.startswith('0X'):
        bytecode_str = bytecode_str[2:]
    
    # Filter out non-hexadecimal characters (newlines, spaces, etc.)
    bytecode_str = re.sub(r'[^0-9a-fA-F]', '', bytecode_str)
    
    if len(bytecode_str) % 2 != 0:
        bytecode_str = bytecode_str[:-1]

    if not bytecode_str:
        return ""

    raw_bytes = bytes.fromhex(bytecode_str)

    tokens = []
    i = 0
    n = len(raw_bytes)

    while i < n:
        byte_val = raw_bytes[i]

        # Check if opcode is PUSH1 .. PUSH32
        if 0x60 <= byte_val <= 0x7F:
            push_num = byte_val - 0x60 + 1
            op_token = f"PUSH{push_num}" if use_mnemonics else f"{byte_val:02x}"
            operand_bytes = raw_bytes[i + 1 : min(i + 1 + push_num, n)]

            if use_opcode_values:
                if use_mnemonics:
                    tokens.append(op_token)
                    if operand_bytes:
                        tokens.append(operand_bytes.hex())
                else:
                    tokens.append(op_token)
                    for b in operand_bytes:
                        tokens.append(f"{b:02x}")
            else:
                tokens.append(op_token)

            i += 1 + push_num
        else:
            if use_mnemonics:
                mnemonic, _ = EVM_OPCODES.get(byte_val, (f"UNKNOWN_{byte_val:02x}", 0))
                tokens.append(mnemonic)
            else:
                tokens.append(f"{byte_val:02x}")
            i += 1

    return " ".join(tokens)
