from transformers import pipeline

summarization = pipeline("summarization")
# Example usage:
original_text = "The State Bank of Vietnam (SBV), in a move to stabilize gold prices, will sell gold bars to four state-owned lenders so they can distribute to retail buyers.\
The sale will begin June 3 at prices determined by the central bank, SBV deputy governor Pham Quang Dung said Wednesday.\
The state-owned lenders involved are Agribank, Vietcombank, BIDV, and VietinBank, which already have extensive networks ready to sell directly to individual buyers, Dung added.\
A representative of one of the banks said that they are not allowed to sell in bulk to organizations and businesses.\
Some private banks have already been selling gold but only at small volume. Most retail consumers buy their gold at jewelry stores.\
This is the latest effort of the central bank in bringing down domestic gold prices after hosting auctions in the last several weeks to increase supply.\
SJC gold prices\
VND million per tael (VND1 million = $39.29)\
Although it has sold 48,500 taels (or 1.8 tons) in nine auctions, Vietnam Saigon Jewelry Company gold bar price is still 20% higher than global rate.\
It has therefore decided cease holding gold auctions.\
Dung said that there are possibilities that speculators have pushed up prices through illegal means.\
A team of inspectors established by the SBV are investigating the gold businesses of major distributors including Saigon Jewelry Company, DOJI and Phu Nhuan Jewelry.\
The central bank has the \"capability and determination\" to stabilize gold prices and the global-domestic price gap will be narrowed sustainably, the deputy governor added.\
Saigon Jewelry Company gold bar price rose by 0.44% to VND90.9 million ($3,571.01) per tael Wednesday morning.\
Globally, gold prices eased on Wednesday, as traders pared bets of rate cuts by the U.S. Federal Reserve this year following remarks by some policymakers, while the market awaited key U.S. inflation data due later this week, Reuters reported.\
Spot gold was down 0.1% at $2,357.70 per ounce. U.S. gold futures were up 0.1% at $2,358.30."
summary = summarization(original_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
print("Generated Summary:", summary[0]["summary_text"])
