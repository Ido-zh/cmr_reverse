# download trained diffusion weights (128x128)
echo "📥 Downloading DDPM weights..."
curl -L https://surfdrive.surf.nl/files/index.php/s/qiclXKFLogKSyuf/download -o dump.zip
if [ $? -ne 0 ]; then
    echo "❌ Downloading failed, exiting."
    exit 1
fi

echo "✅ Downloading successful: dump.zip"
unzip dump.zip
if [ $? -ne 0 ]; then
    echo "❌ unzip failed，exiting"
    exit 1
fi

echo "✅ Unzip successful!"
rm dump.zip
echo "🗑️ Deleting zip."

exit 0

