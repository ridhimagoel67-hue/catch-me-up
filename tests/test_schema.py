from catchmeup.schema import Unit, visible_units

def make(n):
    return [Unit("demo",f"u{i}",i, f"U{i}",f"text {i}","test") 
            for i in range(1,n+1)]

def test_only_earlier_or_equal_units_visible():
    seen=visible_units(make(10), 4)
    assert [u.position for u in seen] == [1,2,3,4]

def test_position_zero_sees_nothing():
    assert visible_units(make(10), 0) == []