from scholar_agent_v2.candidate_pool.models import MatchStatus
from scholar_agent_v2.candidate_pool.tri_logic import tri_and, tri_not, tri_or


def test_tri_and():
    assert tri_and([MatchStatus.PASS, MatchStatus.PASS]) == MatchStatus.PASS
    assert tri_and([MatchStatus.PASS, MatchStatus.UNKNOWN]) == MatchStatus.UNKNOWN
    assert tri_and([MatchStatus.PASS, MatchStatus.FAIL]) == MatchStatus.FAIL
    assert tri_and([]) == MatchStatus.PASS


def test_tri_or():
    assert tri_or([MatchStatus.FAIL, MatchStatus.PASS]) == MatchStatus.PASS
    assert tri_or([MatchStatus.FAIL, MatchStatus.UNKNOWN]) == MatchStatus.UNKNOWN
    assert tri_or([MatchStatus.FAIL, MatchStatus.FAIL]) == MatchStatus.FAIL
    assert tri_or([]) == MatchStatus.FAIL


def test_tri_not():
    assert tri_not(MatchStatus.PASS) == MatchStatus.FAIL
    assert tri_not(MatchStatus.FAIL) == MatchStatus.PASS
    assert tri_not(MatchStatus.UNKNOWN) == MatchStatus.UNKNOWN
