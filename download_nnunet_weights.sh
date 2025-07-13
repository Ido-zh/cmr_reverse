# downloading nnUNet training data
echo "📥 Downloading nnUNet trained weights..."

# find the results folder
if [ -z "$RESULTS_FOLDER" ]; then
    echo "❌ Environment variable RESULTS_FOLDER undefined!"
    exit 1
fi
if [ -d "$RESULTS_FOLDER" ]; then
    echo "✅ Path found: $RESULTS_FOLDER"
else
    echo "❌ Path not found: $RESULTS_FOLDER"
    exit 1
fi

SURFDRIVE_LINK="https://surfdrive.surf.nl/files/index.php/s/aLwZM4htAnBm7ST/download"
EXTRACT_DIR="$RESULTS_FOLDER/nnUNet/2d"
mkdir -p $EXTRACT_DIR
ZIP_NAME="Task900_ACDC_Phys.zip"
ZIP_PATH="$RESULTS_FOLDER/nnUNet/2d/$ZIP_NAME"

curl -L --fail --show-error "$SURFDRIVE_LINK" -o "$ZIP_PATH"

if [ $? -ne 0 ]; then
    echo "❌ download nnUNet training weights failed!"
    exit 1
fi
echo "✅ download nnUNet training weights successful!"

# Unzip
echo "Unzipping..."
mkdir -p "$EXTRACT_DIR"
unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Unzip failed，exiting。"
    exit 1
fi
echo "✅ Unzip successful!: $EXTRACT_DIR"

# delete zip
rm "$ZIP_PATH"
echo "🗑️ Deleted: $ZIP_PATH"

exit 0
