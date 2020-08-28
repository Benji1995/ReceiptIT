import json


# stores = ["Colruyt", "Delhaize", "Aldi", "Lidl", "Albert Heijn", "Jumbo", "Spar", "Carrefour", "Carrefour Express"]
# for s in stores:
#    json_decoded[s]=None

def getStores():
    with open("./JSON_dir/Stores.txt") as json_file:
        json_decoded = json.load(json_file, encoding="utf-8")
    json_file.close()
    return json_decoded, list(json_decoded.keys()), list(json_decoded.values())


def addStore(new_store):
    with open("./JSON_dir/Stores.txt") as json_file:
        json_decoded = json.load(json_file, encoding="utf-8")
    json_file.close()

    json_decoded[new_store] = None

    with open("./JSON_dir/Stores.txt", 'w') as json_file:
        json.dump(json_decoded, json_file, ensure_ascii=False)
    json_file.close()