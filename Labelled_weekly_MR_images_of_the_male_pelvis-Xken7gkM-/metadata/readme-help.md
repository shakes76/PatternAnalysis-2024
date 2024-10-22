# How metadata versions work

Collections in CSIRO's [Data Access Portal](https://data.csiro.au), DAP, are
versioned. A new version may have new or modified files from its previous
version, or it may be a metadata only update. When a collection has had only its
metadata updated, all versions of the metadata and licences are stored with the
original, unchanged files.

Files with a name like `dublin-core-v1.xml` will contain a minimal description
of the DAP collection. You may be able to import this into reference management
software that supports this schema. The files will contain a URL that you can
follow to find the full description of the DAP collection.

It's possible that the changes to the DAP collection metadata between versions
do not involve any of the metadata fields contained in the Dublin Core XML
files, in which case the metadata files will have very similar content. If the
citation details of the DAP collection have changed, or if a different licence
has been specified, these changes will be reflected in the different versions.

# How checksums work

DAP collections will have a text file in the `./metadata` folder containing
[checksums](https://en.wikipedia.org/wiki/Checksum) for all the collection's
files. This file will have a name like
`./metadata/collection_import_sha256sum.txt` where either `sha256sum` or
`sha512sum` is used.

The checksum algorithm used will be part of the file name, in the above example
it is `SHA256`. Each line will consist of a checksum hash value followed by
two spaces and then the path of the file relative to the root folder of the
collection, e.g. one line may looke like:

```
122c577090b4661ea1601f610229aaf72de08c133aeb54a7f00c40bf3e284c49  ./data/164.a2.391.tif
```

These files are designed to work with the Unix `sha256sum` or `sha512sum`
command line tools. Using a terminal you can run a command from the root folder
of the collection to verify the integrity of your downloaded files. If the
checksum value that you generate for a downloaded file is different to the
checksum listed in the text file then the contents of the downloaded file are
different to the original. If this happens you may need to download the file
again.

e.g. when in the parent folder of the `data` and `metadata` folders and you want
to verify checksums created with the `SHA256` algorithm, you can run the
following:

```shell
# Validate all files and print the results
sha256sum -c ./metadata/collection_import_sha256sum.txt
```

```shell
# Validate all files and only print failures
sha256sum -c --quiet ./metadata/collection_import_sha256sum.txt
```

If the name of the checksum file indicates that the `SHA512` algorithm was used,
use the `sha512sum` command and modify the filename in the above example.

Windows does not have an included tool to run the above command. Instead
you could run a script like the following in PowerShell which will give a
similar output to the first Unix example above. This script should also be run
from the parent folder of `data` and `metadata`. You may need to modify the 
values defined in the first two lines of the script.

```powershell
$checksumFileName = "./metadata/collection_import_sha256sum.txt"; ` # Modify this to the actual checksum filename if necessary.
$algorithm = "SHA256"; ` # Modify this to the algorithm used, e.g. "MD5", "SHA512", etc.
$failCount = 0; `
foreach($line in Get-Content $checksumFileName) {
    $components = $line -split ' +', 2
    $remote_checksum = $components[0]
    $file_name = $components[1]
    $local_checksum = Get-FileHash "$file_name" -Algorithm $algorithm | Select-Object -ExpandProperty Hash
    if ($remote_checksum -eq $local_checksum) {
        Write-Output $file_name": OK"
    } else {
        Write-Output $file_name": FAILED"
        $failCount++
    }
} `
if ($failCount -ne 0) {
    if ($failCount -gt 1) {$plural = "s"} else {$plural = ""}
    Write-Warning "$failCount computed checksum$plural did NOT match"
}
```

If you persistently encounter errors validating the checksums then you should
go to the [Data Access Portal](https://data.csiro.au), find the collection you
downloaded, and look for the contact details.
