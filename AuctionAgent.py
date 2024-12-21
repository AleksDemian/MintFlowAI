from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import Wallet

from constant import AUCTION_ABI
from cdp_agentkit_core.actions.deploy_nft import DeployNftAction
from cdp_agentkit_core.actions.mint_nft import MintNftAction

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")  

# Initialize CDP AgentKit wrapper
cdp = CdpAgentkitWrapper()

# Create toolkit from wrapper
cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp)

# Get all available tools
tools = cdp_toolkit.get_tools()

# Add the NFT Auction functionality
START_NFT_AUCTION_PROMPT = """
This tool starts an NFT auction using an existing NFT smart contract. 
It takes the contract address of the NFT, the token ID of the NFT, the starting price, and the duration of the auction as inputs.
The auction will run on-chain, and the highest bidder will win the NFT.
"""

AUCTION_CONTRACT_ADDRESS = "0xA0f0b923532fdbcd85A11ccAED1362691C7931fA"

class DeployNftInput(BaseModel):
    name: str = Field(..., description="The name of the NFT collection, e.g., `MyNFT`", example="MyNFT")
    symbol: str = Field(..., description="The symbol of the NFT collection, e.g., `MNFT`", example="MNFT")
    base_uri: str = Field(..., description="The base URI for the NFT metadata, e.g., `https://example.com/nft/`", example="https://example.com/nft/")

class MintNftInput(BaseModel):
    """Input argument schema for mint NFT action."""

    contract_address: str = Field(
        ...,
        description="The contract address of the NFT (ERC-721) to mint, e.g. `0x036CbD53842c5426634e7929541eC2318f3dCF7e`",
    )
    destination: str = Field(
        ...,
        description="The destination address that will receive the NFT onchain, e.g. `0x036CbD53842c5426634e7929541eC2318f3dCF7e`",
    )

class StartNftAuctionInput(BaseModel):
    """Input argument schema for starting an NFT auction."""

    nft_contract_address: str = Field(
        ...,
        description="The contract address of the NFT, e.g., `0x123abc...`",
        example="0x123abc456def789...",
    )
    token_id: int = Field(
        ...,
        description="The ID of the NFT token to auction, e.g., `1`",
        example=1,
    )
    starting_price: float = Field(
        ...,
        description="The starting price of the auction in ETH, e.g., `0.1`",
        example=0.1,
    )
    duration: int = Field(
        ...,
        description="The duration of the auction in seconds, e.g., `86400` for 1 day",
        example=86400,
    )

class BidOnNftInput(BaseModel):
    token_id: int = Field(
        ...,
        description="The ID of the NFT token to bid on, e.g., `1`",
        example=1,
    )
    bid_amount: float = Field(
        ...,
        description="The amount of ETH to bid, e.g., `0.5`",
        example=0.5,
    )

class FinalizeAuctionInput(BaseModel):
    token_id: int = Field(
        ...,
        description="The ID of the NFT token for the auction to finalize, e.g., `1`",
        example=1,
    )

def start_nft_auction(
    wallet: Wallet,
    nft_contract_address: str,
    token_id: int,
    starting_price: float,
    duration: int
) -> str:
    """
    Start an NFT auction using the deployed auction contract.

    Args:
        wallet (Wallet): The wallet to create the auction from.
        nft_contract_address (str): The address of the NFT contract.
        token_id (int): The ID of the token to be auctioned.
        starting_price (float): The starting price of the auction in ETH.
        duration (int): The duration of the auction in seconds.

    Returns:
        str: A message containing the auction details.
    """
    try:
        starting_price_wei = int(starting_price * 10**18)

        auction_args = {
            "nftContract": nft_contract_address,
            "tokenId": token_id,
            "startingPrice": starting_price_wei,
            "duration": duration,
        }

        auction_creation = wallet.invoke_contract(
            contract_address=AUCTION_CONTRACT_ADDRESS,  
            method="createAuction",
            args=auction_args,
            abi=AUCTION_ABI, 
        ).wait()

        return (
            f"Auction created successfully for token ID {token_id}.\n"
            f"NFT Contract: {nft_contract_address}\n"
            f"Starting price: {starting_price} ETH\n"
            f"Duration: {duration} seconds\n"
            f"Transaction hash: {auction_creation.transaction.transaction_hash}\n"
            f"Transaction link: {auction_creation.transaction.transaction_link}"
        )

    except Exception as e:
        return f"Error creating auction: {str(e)}"

def bid_on_nft(wallet: Wallet, token_id: int, bid_amount: float) -> str:
    """
    Place a bid on an NFT auction.

    Args:
        wallet (Wallet): The wallet to place the bid from.
        token_id (int): The token ID of the NFT to bid on.
        bid_amount (float): The amount to bid in ETH.

    Returns:
        str: A message containing the bid details.
    """
    try:
        bid_args = {"tokenId": token_id}

        bid_result = wallet.invoke_contract(
            contract_address=AUCTION_CONTRACT_ADDRESS,
            method="bid",
            args=bid_args,
            abi=AUCTION_ABI,
            value=int(bid_amount * 10**18), 
        ).wait()

        return (
            f"Bid placed on token ID {token_id} with amount {bid_amount} ETH.\n"
            f"Transaction hash: {bid_result.transaction.transaction_hash}\n"
            f"Transaction link: {bid_result.transaction.transaction_link}"
        )

    except Exception as e:
        return f"Error placing bid: {str(e)}"

def finalize_nft_auction(wallet: Wallet, token_id: int) -> str:
    """
    Finalize an NFT auction.

    Args:
        wallet (Wallet): The wallet to finalize the auction from.
        token_id (int): The token ID of the NFT to finalize.

    Returns:
        str: A message containing the auction finalization details.
    """
    try:
        finalize_args = {"tokenId": token_id}

        finalize_result = wallet.invoke_contract(
            contract_address=AUCTION_CONTRACT_ADDRESS,
            method="finalizeAuction",
            args=finalize_args,
            abi=AUCTION_ABI,
        ).wait()

        return (
            f"Auction finalized for token ID {token_id}.\n"
            f"Transaction hash: {finalize_result.transaction.transaction_hash}\n"
            f"Transaction link: {finalize_result.transaction.transaction_link}"
        )

    except Exception as e:
        return f"Error finalizing auction: {str(e)}"

startNftAuctionTool = CdpTool(
    name="start_nft_auction",
    description=START_NFT_AUCTION_PROMPT,
    cdp_agentkit_wrapper=cdp,
    args_schema=StartNftAuctionInput,
    func=start_nft_auction,
)

bidOnNftTool = CdpTool(
    name="bid_on_nft",
    description="Place a bid on an NFT auction.",
    cdp_agentkit_wrapper=cdp,
    args_schema=BidOnNftInput, 
    func=bid_on_nft,
)

finalizeAuctionTool = CdpTool(
    name="finalize_nft_auction",
    description="Finalize an NFT auction and transfer the token to the highest bidder.",
    cdp_agentkit_wrapper=cdp,
    args_schema=FinalizeAuctionInput, 
    func=finalize_nft_auction,
)

deployNftTool = CdpTool(
    name="deploy_nft",
    description="Deploy an ERC-721 NFT contract with a given name, symbol, and base URI.",
    cdp_agentkit_wrapper=cdp,
    args_schema=DeployNftInput,
    func=DeployNftAction().func,  
)

mintNftTool = CdpTool(
    name="mint_nft",
    description="Mint an NFT from an existing ERC-721 contract.",
    cdp_agentkit_wrapper=cdp,
    args_schema=MintNftInput,
    func=MintNftAction().func,  
)

tools.extend([startNftAuctionTool, bidOnNftTool, finalizeAuctionTool, deployNftTool, mintNftTool])

# Create the agent
agent_executor = create_react_agent(
    llm,
    tools=tools,
    state_modifier="You are a helpful agent that can interact with the Base blockchain using CDP AgentKit. You can create wallets, deploy tokens, and perform transactions."
)

# Function to interact with the agent
def ask_agent(question: str):
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "my_first_agent"}}
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            print(chunk["tools"]["messages"][0].content)
        print("-------------------")

# Test the agent
if __name__ == "__main__":
    print("Agent is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        ask_agent(user_input)