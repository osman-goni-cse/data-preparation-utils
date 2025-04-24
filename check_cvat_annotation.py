# check_cvat_annotation.py
# How to use: python3 check_cvat_annotation.py 
# Checking Rules:
# 1. CN: 4 Cap letters + 7 numbers (ABCD1234567)
# 2. CN_ABC: 4 Cap letters (ABCD)
# 3. CN_NUM: 7 numbers (1234567)
# 4. TS: 1 any + 1 number + 1 any + 1 number (22G1, L5G1 etc.)
# 5. C_DIGIT: 1 number
# 6. CN_ABC = CN[:4] (first 4 digits)
# 7. CN_NUM = CN[-7:] (last 7 digits)
# 8. CN check digit calculation should be correct && CN[-1:] = C_DIGIT

import os
import xml.etree.ElementTree as ET
import re
from check_digit_calculation import calculate_check_digit
import argparse

def validate_annotations(xml_file):
    # Load and parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define the validation rules as regular expressions, rule 1-5
    rules = {
        "CN": {"attribute": "cn_text", "regex": r"^[A-Z]{4}\d{7}$"},
        "CN_ABC": {"attribute": "cn_abc_text", "regex": r"^[A-Z]{4}$"},
        "CN_NUM": {"attribute": "cn_num_text", "regex": r"^\d{7}$"},
        "TS": {"attribute": "ts_text", "regex": r"^.(\d).(\d)$"},
        "C_DIGIT": {"attribute": "c_digit_num", "regex": r"^\d$"}
    }

    background_img = 0

    # Iterate through images and their boxes
    for image in root.findall('image'):
        image_name = image.get('name')
        labels_attributes = {}

        if not image.findall('box'):
            background_img += 1

        # Collect all attributes for each label
        for box in image.findall('box'):
            label = box.get('label')
            if "rotation" in box.attrib:
                attribute_name = rules[label]["attribute"]
                attribute_element = box.find(f"attribute[@name='{attribute_name}']")
                value = attribute_element.text
                print(f"Image: {image_name}, Label: {label}, Value: {value} Rotation Exists")

        #     for attribute in box.findall('attribute'):
        #         attribute_name = attribute.get('name')
        #         attribute_value = attribute.text
        #         if label not in labels_attributes:
        #             labels_attributes[label] = {}
        #         labels_attributes[label][attribute_name] = attribute_value
        # # print(image_name)
        # # print(labels_attributes)
        # # Validate each label and attribute
        # for label, attributes in labels_attributes.items():
        #     for attribute_name, attribute_value in attributes.items():
        #         if label in rules and rules[label]["attribute"] == attribute_name:
        #             # if not re.match(rules[label]["regex"], attribute_value):
        #             #     print(f"Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")
        #             try:
        #                 if not re.match(rules[label]["regex"], attribute_value):
        #                     print(f"Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")
        #             except TypeError as e:
        #                 print(f"TypeError: {e} - Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")
        #             except re.error as e:
        #                 print(f"Regex error: {e} - Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")

        #         # rule 8
        #         if label == "CN" and attribute_name == "cn_text":
        #             if len(attribute_value) != 11: # CN should be 11 digits
        #                 print(f"Image: {image_name}, Label: CN, Invalid length: {attribute_value}")
        #             else:
        #                 expected_check_digit = calculate_check_digit(attribute_value[:10])
        #                 actual_check_digit = int(attribute_value[10])
        #                 if expected_check_digit != actual_check_digit:
        #                     print(f"Image: {image_name}, Label: CN, Invalid check digit in: {attribute_value}")
        # # rule 6
        # if "CN" in labels_attributes and "CN_ABC" in labels_attributes:
        #     if labels_attributes["CN"].get("cn_text")[:4] != labels_attributes["CN_ABC"].get("cn_abc_text"):
        #         print(f"Image: {image_name}, Label: CN and CN_ABC, Mismatch in first 4 characters")
        # # rule 7
        # if "CN" in labels_attributes and "CN_NUM" in labels_attributes:
        #     if labels_attributes["CN"].get("cn_text")[-7:] != labels_attributes["CN_NUM"].get("cn_num_text"):
        #         print(f"Image: {image_name}, Label: CN and CN_NUM, Mismatch in last 7 digits")
            for attribute in box.findall('attribute'):
                attribute_name = attribute.get('name')
                attribute_value = attribute.text
                if label not in labels_attributes:
                    labels_attributes[label] = {}
                if attribute_name not in labels_attributes[label]:
                    labels_attributes[label][attribute_name] = []
                labels_attributes[label][attribute_name].append(attribute_value)

        # if(image.get('id') == '1014'):
        #     print(image_name)
        #     print(labels_attributes)

        # Validate each label and attribute
        for label, attributes in labels_attributes.items():
            for attribute_name, attribute_values in attributes.items():
                for attribute_value in attribute_values:
                    if label in rules and rules[label]["attribute"] == attribute_name:
                        try:
                            if not re.match(rules[label]["regex"], attribute_value):
                                print(f"Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")
                        except TypeError as e:
                            print(f"TypeError: {e} - Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")
                        except re.error as e:
                            print(f"Regex error: {e} - Image: {image_name}, Label: {label}, Attribute: {attribute_name}, Invalid value: {attribute_value}")

                    # rule 8
                    # if label == "CN" and attribute_name == "cn_text":
                    #     if len(attribute_value) != 11: # CN should be 11 digits
                    #         print(f"Image: {image_name}, Label: CN, Invalid length: {attribute_value}")
                    #     else:
                    #         expected_check_digit = calculate_check_digit(attribute_value[:10])
                    #         actual_check_digit = int(attribute_value[10])
                    #         if expected_check_digit != actual_check_digit:
                    #             print(f"Image: {image_name}, Label: CN, Invalid check digit in: {attribute_value}")
                    if label == "CN" and attribute_name == "cn_text":
                        try:
                            if len(attribute_value) != 11:  # CN should be 11 digits
                                print(f"Image: {image_name}, Label: CN, Invalid length: {attribute_value}")
                            else:
                                try:
                                    expected_check_digit = calculate_check_digit(attribute_value[:10])
                                    actual_check_digit = int(attribute_value[10])
                                    if expected_check_digit != actual_check_digit:
                                        print(f"Image: {image_name}, Label: CN, Invalid check digit in: {attribute_value}")
                                except ValueError as e:
                                    print(f"ValueError: {e} - Unable to convert check digit to integer in: {attribute_value}")
                                except Exception as e:
                                    print(f"Unexpected error during check digit calculation: {e} - Value: {attribute_value}")
                        except TypeError as e:
                            print(f"TypeError: {e} - Attribute value is not a valid string or None: {attribute_value}")
                        except Exception as e:
                            print(f"Unexpected error: {e} - Attribute value: {attribute_value}")

        # rule 6
        # if "CN" in labels_attributes and "CN_ABC" in labels_attributes:
        #     cn_texts = labels_attributes["CN"].get("cn_text", [])
        #     cn_abc_texts = labels_attributes["CN_ABC"].get("cn_abc_text", [])
        #     mismatch_found = True
        #     for cn_text in cn_texts:
        #         for cn_abc_text in cn_abc_texts:
        #             if cn_text[:4] == cn_abc_text:
        #                 mismatch_found = False
        #                 break
        #         if not mismatch_found:
        #             break
        #     if mismatch_found:
        #         print(f"Image: {image_name}, Label: CN and CN_ABC, Mismatch in first 4 characters")
        if "CN" in labels_attributes and "CN_ABC" in labels_attributes:
            try:
                cn_texts = labels_attributes["CN"].get("cn_text", [])
                cn_abc_texts = labels_attributes["CN_ABC"].get("cn_abc_text", [])
                mismatch_found = True
                for cn_text in cn_texts:
                    for cn_abc_text in cn_abc_texts:
                        try:
                            if cn_text[:4] == cn_abc_text:
                                mismatch_found = False
                                break
                        except TypeError as e:
                            print(f"TypeError: {e} - Invalid data type for CN or CN_ABC text. CN: {cn_text}, CN_ABC: {cn_abc_text}")
                        except Exception as e:
                            print(f"Unexpected error: {e} - CN: {cn_text}, CN_ABC: {cn_abc_text}")
                    if not mismatch_found:
                        break
                if mismatch_found:
                    print(f"Image: {image_name}, Label: CN and CN_ABC, Mismatch in first 4 characters")
            except KeyError as e:
                print(f"KeyError: {e} - Missing key in labels_attributes")
            except Exception as e:
                print(f"Unexpected error: {e} - While processing CN and CN_ABC labels")

        # rule 7
        # if "CN" in labels_attributes and "CN_NUM" in labels_attributes:
        #     cn_texts = labels_attributes["CN"].get("cn_text", [])
        #     cn_num_texts = labels_attributes["CN_NUM"].get("cn_num_text", [])
        #     mismatch_found = True
        #     for cn_text in cn_texts:
        #         for cn_num_text in cn_num_texts:
        #             if cn_text[-7:] == cn_num_text:
        #                 mismatch_found = False
        #                 break
        #         if not mismatch_found:
        #             break
        #     if mismatch_found:
        #         print(f"Image: {image_name}, Label: CN and CN_NUM, Mismatch in last 7 digits")
        if "CN" in labels_attributes and "CN_NUM" in labels_attributes:
            try:
                cn_texts = labels_attributes["CN"].get("cn_text", [])
                cn_num_texts = labels_attributes["CN_NUM"].get("cn_num_text", [])
                mismatch_found = True
                for cn_text in cn_texts:
                    for cn_num_text in cn_num_texts:
                        try:
                            if cn_text[-7:] == cn_num_text:
                                mismatch_found = False
                                break
                        except TypeError as e:
                            print(f"TypeError: {e} - Invalid data type for CN or CN_NUM text. CN: {cn_text}, CN_NUM: {cn_num_text}")
                        except Exception as e:
                            print(f"Unexpected error: {e} - CN: {cn_text}, CN_NUM: {cn_num_text}")
                    if not mismatch_found:
                        break
                if mismatch_found:
                    print(f"Image: {image_name}, Label: CN and CN_NUM, Mismatch in last 7 digits")
            except KeyError as e:
                print(f"KeyError: {e} - Missing key in labels_attributes")
            except Exception as e:
                print(f"Unexpected error: {e} - While processing CN and CN_NUM labels")
    
    print(f"Background images: {background_img}")

import zipfile
import tempfile

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="validate raw annotation container number")
    parser.add_argument("--zip_file_path", default="/home/osman/Downloads/export-data")

    args = parser.parse_args()
    # raw_cvat_annotations_folder = args.zip_file_path # replace with your absolute folder path

    # for filename in os.listdir(raw_cvat_annotations_folder): # check all the annotation files
    #   if filename.endswith(".xml"):
    #       print(filename + " check start.")
    #       validate_annotations(os.path.join(raw_cvat_annotations_folder, filename))
    #       print(filename + " check done.")

    with zipfile.ZipFile(args.zip_file_path, 'r') as zip_ref:
        if "annotations.xml" in zip_ref.namelist():
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extract("annotations.xml", temp_dir)
                xml_file_path = os.path.join(temp_dir, "annotations.xml")
                print("annotations.xml check start.")
                validate_annotations(xml_file_path)
                print("annotations.xml check done.")
