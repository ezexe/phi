"""The workflows run on GitHub's runners as they are: every action they use sits at a major that declares Node 24."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ("tests.yml", "pages.yml")

# The first major of each action whose action.yml declares node24, or, for the composite upload-pages-artifact, whose
# upload-artifact does, read from https://raw.githubusercontent.com/actions/<action>/<tag>/action.yml at each tag.
FIRST_NODE24_MAJOR = {
    "actions/checkout": 5,
    "actions/setup-python": 6,
    "actions/upload-artifact": 6,
    "actions/upload-pages-artifact": 5,
    "actions/deploy-pages": 5,
    "actions/configure-pages": 6,
}


def test_every_action_sits_at_a_major_that_declares_node24():
    """GitHub removed Node 20 from its runners on 2026-09-23, and runs an action that declares node20 on Node 24 with a
    warning on every job."""
    pins = []
    for name in WORKFLOWS:
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        pins += [(name, action, int(major)) for action, major in re.findall(r"uses:\s*([\w.-]+/[\w.-]+)@v(\d+)", text)]
    assert pins, "the workflows name no action"
    for name, action, major in pins:
        assert action in FIRST_NODE24_MAJOR, f"{name}: {action} has no first node24 major in this test"
        assert major >= FIRST_NODE24_MAJOR[action], f"{name}: {action}@v{major} declares a Node before 24"
