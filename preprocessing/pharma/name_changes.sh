#!/usr/bin/env bash
# author: stephanie

# find out how many drugs (identified by pharmaID) changed name

# field numbers:
#1 ArchStatus
#2 ArchTime
#3 Availability
#4 BgColorRGB
#5 BloodFlag
#6 BuyPrice
#7 CanBeMixed
#8 ControlledInOrder
#9 ControlledInRecord
#10 DefaultRoute
#11 DoseFormRatio
#12 DoseUnitID
#13 ExtCode
#14 Form
#15 FormUnitID
#16 GenericName
#17 InFluidFormRatio
#18 Infusable
#19 PerNUnits
#20 PharmaID
#21 PharmaName
#22 PharmaStatus
#23 SellPrice
#24 Type
#25 VolumeFormRatio
#26 VolumeUnitID

# grab pharmIDs with two pharmaNames (fields 20, 21)
cut -f 20,21 -d$'\t'  expot-pharmaref.csv | sort -u > name_pharmaID.steph_tmp
awk 'FS="\t"{ print $1 }' name_pharmaID.steph_tmp | sort | uniq -c | grep -v ' 1 ' | awk '{ print $2 }' | sort -u  > twonames.steph_tmp

# intersect with the pharma IDs in the excel file
comm -12 excel_pharmaIDs.txt twonames.steph_tmp > uh.tmp
mv uh.tmp twonames.steph_tmp

# now get their names
while read pharmaid;
do
    grep "^"$pharmaid"	" name_pharmaID.steph_tmp >> twonames_2.steph_tmp
done < twonames.steph_tmp

# sort, etc.
sort -u twonames_2.steph_tmp  | sort -n > uh.steph_tmp

# how many duplicates?
n_dupes=`wc -l twonames.steph_tmp | awk '{ print $1 }'`
echo "# pharmaIDs with >1 PharmaName:" $n_dupes

# clean up
mv uh.steph_tmp pharma_multiple_names.txt
rm -v *steph_tmp
