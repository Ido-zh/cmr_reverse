# downloading nnUNet training data
echo "📥 Downloading nnUNet training data..."

# find the nnUNet_raw_data_base
if [ -z "$nnUNet_raw_data_base" ]; then
    echo "❌ Environment variable nnUNet_raw_data_base undefined!"
    exit 1
fi
if [ -d "$nnUNet_raw_data_base" ]; then
    echo "✅ Path found: $nnUNet_raw_data_base"
else
    echo "❌ Path not found: $nnUNet_raw_data_base"
    exit 1
fi

SURFDRIVE_LINK="https://surfdrive.surf.nl/files/index.php/s/SZQbuO7dc9HX7av/download"
EXTRACT_DIR="$nnUNet_raw_data_base/nnUNet_raw_data"
ZIP_NAME="Task900_ACDC_Phys.zip"
ZIP_PATH="$nnUNet_raw_data_base/nnUNet_raw_data/$ZIP_NAME"

curl -L --fail --show-error "$SURFDRIVE_LINK" -o "$ZIP_PATH"

if [ $? -ne 0 ]; then
    echo "❌ download nnUNet training data failed!"
    exit 1
fi
echo "✅ download nnUNet training data successful!"

#  Unzip
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
