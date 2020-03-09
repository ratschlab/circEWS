#!/usr/bin/env bash

cut -d$'\t' -f 12,20 expot-pharmaref.csv | sort -u > unit_pharmaID.steph_tmp
awk '{ print $2 }' unit_pharmaID.steph_tmp | sort | uniq -c | grep -v ' 1 ' | awk '{ print $2 }' | sort > multiple_units.steph_tmp

# intersect with pharma IDs in the excel file

comm -12 excel_pharmaIDs.txt multiple_units.steph_tmp > uh.steph_tmp
mv uh.steph_tmp multiple_units.steph_tmp

# add on the unit IDs
while read unitid;
do
    grep "	"$unitid"$" unit_pharmaID.steph_tmp >> multiple_units_2.steph_tmp
done < multiple_units.steph_tmp

# clean up
echo "UnitID	PharmaID" > header.steph_tmp
cat header.steph_tmp multiple_units_2.steph_tmp > multiple_units.txt
rm -v *.steph_tmp
