

def test_mbpp_item_takes_function_from_first_assert():
    from vigia.datasets import mbpp_item

    item = mbpp_item({"task_id": 2, "text": "Sumar dos números.", "test_setup_code": "",
                      "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]})
    assert item["id"] == "MBPP/2" and item["entry_point"] == "add"
    assert "assert add(1, 2) == 3" in item["question"] and item["tests"].count("assert") == 2
