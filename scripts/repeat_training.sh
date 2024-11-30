#!/bin/bash

# ANSI color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to add days to a date
add_days() {
    date -d "$1 + $2 days" +%Y-%m-%d
}

# Function to print separator
print_separator() {
    echo -e "${BLUE}================================================${NC}"
}

# Generate dates with 6-day intervals
start_date="2024-09-01"
end_date="2024-10-20"
current_date=$start_date
end_dates=()

# Generate all prediction dates
while [[ "$current_date" < "$end_date" ]]; do
    end_dates+=("$current_date")
    current_date=$(add_days "$current_date" 6)
done

# Add the final end date if it's not already included
if [[ "${end_dates[-1]}" != "$end_date" ]]; then
    end_dates+=("$end_date")
fi

# Print prediction schedule
print_separator
echo -e "${GREEN}Prediction Schedule:${NC}"
print_separator
echo -e "${YELLOW}Start Date: $start_date${NC}"
echo -e "${YELLOW}Final Date: $end_date${NC}"
echo -e "\nPrediction will be made for the following dates:"
for (( i=0; i<${#end_dates[@]}; i++ )); do
    echo -e "${GREEN}[$((i+1))/${#end_dates[@]}]${NC} ${end_dates[i]}"
done
print_separator

# Confirm execution
echo -e "\nTotal number of predictions to run: ${GREEN}${#end_dates[@]}${NC}"
read -p "Press Enter to start predictions or Ctrl+C to cancel..."

# Loop through each end date
for (( i=0; i<${#end_dates[@]}; i++ )); do
    end_date=${end_dates[i]}
    print_separator
    echo -e "${GREEN}[$((i+1))/${#end_dates[@]}] Running prediction for end date: $end_date${NC}"
    print_separator
    
    # Run the prediction script with the current end date
    python training/fine_tune_v2.py stock_info.end_date=$end_date
    
    echo -e "\n${YELLOW}Finished prediction for end date: $end_date${NC}"
    print_separator
done

echo -e "\n${GREEN}All predictions completed successfully!${NC}"
print_separator

# Print summary
echo -e "\n${GREEN}Summary of Predictions:${NC}"
echo -e "Start Date: $start_date"
echo -e "End Date: $end_date"
echo -e "Total Predictions: ${#end_daztes[@]}"
echo -e "Prediction Dates:"
for (( i=0; i<${#end_dates[@]}; i++ )); do
    echo -e "${GREEN}[$((i+1))/${#end_dates[@]}]${NC} ${end_dates[i]}"
done
print_separator