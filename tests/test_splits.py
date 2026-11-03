import pandas as pd
import pytest

from whoisit.data.splits import make_split, pick_photos


def fake_df():
    # 100 people with 25 photos, 20 people with only 5 photos
    rows = []
    for person in range(120):
        n = 25 if person < 100 else 5
        for i in range(n):
            rows.append((f"{person}_{i}.jpg", person))
    return pd.DataFrame(rows, columns=["image", "person"])


def small_split(df, seed=42):
    return make_split(df, seed=seed, n_train=50, n_val=20, n_test=30)


def test_same_seed_same_split():
    df = fake_df()
    a = small_split(df)
    b = small_split(df)
    for x, y in zip(a, b):
        assert list(x) == list(y)


def test_different_seed_other_split():
    df = fake_df()
    a = small_split(df, seed=1)
    b = small_split(df, seed=2)
    assert list(a[0]) != list(b[0])


def test_no_person_in_two_groups():
    train, val, test = small_split(fake_df())
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)


def test_sizes_and_few_photos_left_out():
    train, val, test = small_split(fake_df())
    assert (len(train), len(val), len(test)) == (50, 20, 30)
    assert max(max(train), max(val), max(test)) < 100  # people with 5 photos are out


def test_pick_photos():
    df = fake_df()
    train, val, test = small_split(df)
    photos = pick_photos(df, test, n=10)
    assert len(photos) == 30 * 10
    assert (photos.groupby("person").size() == 10).all()
    assert (photos["image"].str.split("_").str[0].astype(int) == photos["person"]).all()


def test_too_few_people():
    with pytest.raises(ValueError):
        make_split(fake_df(), n_train=80, n_val=20, n_test=30)