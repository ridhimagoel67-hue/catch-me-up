from catchmeup.adapters.fandom import clean_wikitext

def test_links_and_markup_are_removed():
    raw = "==Recap==\nBlair meets [[Serena van der Woodsen|Serena]] at '''school'''.{{Quote|x}}"
    assert clean_wikitext(raw) == "Blair meets Serena at school."


def test_images_are_removed():
    raw = "Hello [[File:a.jpg|thumb|caption]] world"
    assert clean_wikitext(raw) == "Hello world"