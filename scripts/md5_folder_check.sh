echo "Usage: $0 <folder_path>"
folder_path="$1"

if [ ! -d "$folder_path" ]; then
    echo "Error: $folder_path is not a directory"
    exit 1
fi

find "$folder_path" -type f -iname "*.pdf" -print0 \
| xargs -0 md5sum \
| sort \
| awk '
{
    hash=$1
    file=$2
    files[hash]=files[hash] "\n" file
    count[hash]++
}
END {
    for (h in count)
        if (count[h] > 1)
            print "MD5:", h, files[h] "\n"
}'