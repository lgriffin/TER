"""L3 grounded detectors: the change surface (TER-EVD-006) and architecture
boundary violations (TER-EVD-007), on a synthetic repository committed with
Git (``ter4_shop_repo``) and synthetic sessions: positive, negative and
boundary cases, plus the rules they rest on."""

from __future__ import annotations

from pathlib import Path

import pytest
from ter4_lean_builder import PASS, Script
from ter4_shop_repo import SHOP, at, shop_repo

from ter.adapters.driven.import_linter import ImportLinterContracts
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.application.ground import ground_session
from ter.bootstrap.capabilities import repository_evidence
from ter.domain import AnalysisEngine, explain_batch
from ter.domain.lean import LeanAnalysis, build_countermeasures, explain
from ter.domain.lean.countermeasures import ActionKind, follow_ups
from ter.domain.lean.grounding import RepositoryGrounding, distinctive, replay_edit
from ter.domain.lean.model import Finding, FindingKind, LeanWaste
from ter.domain.lean.surface import (
    GROUNDED_DETECTORS,
    EditPlacement,
    SeedBasis,
)
from ter.domain.repository import (
    ArchitectureContract,
    ContractFormatError,
    ContractKind,
    ImportEdge,
    Layer,
    contract_violations,
    imported_modules,
    repository_path,
    session_root,
)

REPO = Path(__file__).resolve().parents[2]
GROUNDED = {d.id for d in GROUNDED_DETECTORS}

PRICING = "src/app/domain/pricing.py"
MODEL = "src/app/domain/model.py"
CHECKOUT = "src/app/service/checkout.py"
WEB = "src/app/adapters/web.py"
SUMMARY = "src/app/reports/summary.py"


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    return shop_repo(tmp_path / "shop")


def ground(
    script: Script, root: Path, engine: str = "python-ast", contracts: bool = True
) -> RepositoryGrounding:
    return ground_session(
        script.events,
        repository_evidence(root, engine),
        ImportLinterContracts() if contracts else None,
    )


def analyse(
    script: Script, root: Path, engine: str = "python-ast", contracts: bool = True
) -> LeanAnalysis:
    g = ground(script, root, engine, contracts)
    return explain(script.events, RegexTokenizer(), repository=g)


def found(analysis: LeanAnalysis, detector: str) -> list[Finding]:
    return [f for f in analysis.findings if f.detector == detector]


def pricing_task(s: Script, prompt: str = "Fix the rounding in pricing.py") -> None:
    s.prompt(prompt)
    s.read(at(PRICING), SHOP[PRICING])
    s.edit(at(PRICING), "* 1.2", "* 12 / 10")


# --- rules the surface rests on --------------------------------------------


@pytest.mark.req("TER-EVD-006")
class TestSessionPaths:
    def test_the_root_is_the_prefix_most_paths_agree_on(self) -> None:
        files = {"src/a.py", "src/b.py", "README.md"}
        paths = ["/w/p/src/a.py", "/w/p/src/b.py", "/tmp/x.py", "/w/p/README.md"]
        assert session_root(paths, files) == "/w/p"
        assert session_root(["relative/a.py", "/elsewhere/z.py"], files) is None

    def test_a_session_that_only_creates_files_is_placed_by_directory(
        self,
    ) -> None:
        files = {"src/pkg/a.py", "README.md"}
        assert session_root(["/w/p/src/pkg/new.py"], files) == "/w/p"
        assert session_root(["/w/p/elsewhere/new.py"], files) is None

    def test_paths_map_under_the_root_and_not_outside_it(self) -> None:
        assert repository_path("/w/p/src/new.py", "/w/p") == "src/new.py"
        assert repository_path("/w/pq/src/a.py", "/w/p") is None
        assert repository_path("/w/p", "/w/p") is None
        assert repository_path("/tmp/x.py", "/w/p") is None
        assert repository_path("/w/p/a.py", None) is None
        assert repository_path("./src/a.py", None) == "src/a.py"
        assert repository_path("../a.py", "/w/p") is None


@pytest.mark.req("TER-EVD-006")
class TestNamesAndReplay:
    def test_distinctive_symbols_can_be_told_from_prose(self) -> None:
        assert distinctive("tests_importing") and distinctive("ExplainSession")
        assert distinctive("MAX_SIZE")
        assert not any(distinctive(n) for n in ("run", "main", "Calc", "__init__"))

    def test_an_edit_replays_on_the_known_text(self) -> None:
        assert replay_edit("a b a", {"old_string": "a", "new_string": "c"}) == "c b a"
        every = {"old_string": "a", "new_string": "c", "replace_all": True}
        assert replay_edit("a b a", every) == "c b c"
        assert replay_edit("x", {"content": "new"}) == "new"
        multi = {
            "edits": [
                {"old_string": "a", "new_string": "b"},
                {"old_string": "b b", "new_string": "c"},
            ]
        }
        assert replay_edit("a b", multi) == "c"

    def test_an_edit_that_cannot_apply_makes_the_text_unknown(self) -> None:
        assert replay_edit("abc", {"old_string": "zz", "new_string": "y"}) is None
        assert replay_edit(None, {"old_string": "a", "new_string": "b"}) is None
        assert replay_edit(None, {"old_string": "", "new_string": "new"}) == "new"
        assert replay_edit("full", {"old_string": "", "new_string": "x"}) is None
        assert replay_edit("a", {"file_path": "x"}) is None


@pytest.mark.req("TER-EVD-007")
class TestImportedModules:
    def test_a_from_import_of_a_submodule_depends_on_the_submodule(self) -> None:
        modules = {"pkg", "pkg.core"}
        assert imported_modules(ImportEdge("pkg", ("core",), 1), modules) == (
            "pkg.core",
        )
        assert imported_modules(ImportEdge("pkg", ("core", "name"), 1), modules) == (
            "pkg",
            "pkg.core",
        )
        assert imported_modules(ImportEdge("pkg.core", ("add",), 1), modules) == (
            "pkg.core",
        )
        assert imported_modules(ImportEdge("os.path", (), 1), modules) == ("os.path",)
        assert imported_modules(ImportEdge("..up", ("x",), 1), modules) == ()


def _layers(*layers: Layer, containers: tuple[str, ...] = ()) -> ArchitectureContract:
    return ArchitectureContract(
        "L", "layers", ContractKind.LAYERS, layers=layers, containers=containers
    )


@pytest.mark.req("TER-EVD-007")
class TestContractRules:
    def test_a_lower_layer_must_not_import_a_higher_one(self) -> None:
        c = _layers(Layer(("a.top",)), Layer(("a.mid",)), Layer(("a.low",)))
        [v] = contract_violations("a.low.x", "a.top.y", [c])
        assert v.contract == "L" and "a.low is a lower layer than a.top" in v.rule
        assert contract_violations("a.top.y", "a.low.x", [c]) == ()
        assert contract_violations("a.mid", "a.mid.z", [c]) == ()
        assert contract_violations("other", "a.top", [c]) == ()

    def test_independent_siblings_must_not_import_each_other(self) -> None:
        c = _layers(Layer(("a.top",)), Layer(("a.left", "a.right"), independent=True))
        assert len(contract_violations("a.left", "a.right.x", [c])) == 1
        loose = _layers(Layer(("a.top",)), Layer(("a.left", "a.right")))
        assert contract_violations("a.left", "a.right.x", [loose]) == ()

    def test_containers_scope_the_layers(self) -> None:
        c = _layers(Layer(("ui",)), Layer(("core",)), containers=("p1", "p2"))
        assert len(contract_violations("p2.core.m", "p2.ui", [c])) == 1
        assert contract_violations("p1.core", "p2.ui", [c]) == ()

    def test_forbidden_and_independence(self) -> None:
        forbidden = ArchitectureContract(
            "F",
            "f",
            ContractKind.FORBIDDEN,
            source_modules=("a.domain",),
            forbidden_modules=("requests",),
        )
        assert len(contract_violations("a.domain.m", "requests.api", [forbidden])) == 1
        assert contract_violations("a.domain.m", "requests_mock", [forbidden]) == ()
        independent = ArchitectureContract(
            "I", "i", ContractKind.INDEPENDENCE, modules=("a.x", "a.y")
        )
        assert len(contract_violations("a.x.m", "a.y", [independent])) == 1
        assert contract_violations("a.x.m", "a.x.n", [independent]) == ()

    def test_ignored_imports_are_exempt_with_wildcards(self) -> None:
        c = ArchitectureContract(
            "F",
            "f",
            ContractKind.FORBIDDEN,
            source_modules=("a",),
            forbidden_modules=("b",),
            ignore_imports=("a.*.cli -> b", "a.deep.** -> b.**"),
        )
        assert contract_violations("a.x.cli", "b", [c]) == ()
        assert len(contract_violations("a.x.y.cli", "b", [c])) == 1
        assert contract_violations("a.deep.p.q", "b.r.s", [c]) == ()
        assert len(contract_violations("a.deep", "b.r", [c])) == 1


@pytest.mark.req("TER-EVD-007")
class TestImportLinterReader:
    def test_reads_this_repositorys_own_contracts(self) -> None:
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        contracts = ImportLinterContracts().read("pyproject.toml", text)
        by_id = {c.id: c for c in contracts}
        assert {"hexagon-layers", "pure-domain", "independent-adapters"} <= set(by_id)
        layers = by_id["hexagon-layers"]
        assert layers.kind is ContractKind.LAYERS
        assert layers.layers[0] == Layer(("ter.bootstrap",))
        assert layers.layers[-1] == Layer(("ter.domain",))
        assert by_id["independent-adapters"].kind is ContractKind.INDEPENDENCE
        assert all(c.source == "pyproject.toml" for c in contracts)
        # Evaluated on this repository's real rules:
        [v] = contract_violations(
            "ter.domain.lean.model", "ter.ports.driven", contracts
        )
        assert v.contract == "hexagon-layers"
        assert (
            contract_violations("ter_calculator.cli", "ter.bootstrap", contracts) == ()
        )
        assert [
            v.contract
            for v in contract_violations(
                "ter_calculator.compute", "ter.bootstrap", contracts
            )
        ] == ["ter3-uses-hexagon-edges"]

    def test_reads_ini_configuration_and_skips_types_it_has_no_rule_for(self) -> None:
        text = (
            "[importlinter]\nroot_package = app\n\n"
            "[importlinter:contract:layers]\nname = Layers\ntype = layers\n"
            "layers =\n    app.ui\n    (app.api | app.cli)\n    app.core : app.util\n\n"
            "[importlinter:contract:custom]\nname = Custom\n"
            "type = myproject.contracts.CustomContract\nmodules = app\n"
        )
        [c] = ImportLinterContracts().read(".importlinter", text)
        assert c.id == "layers" and c.name == "Layers"
        assert c.layers == (
            Layer(("app.ui",)),
            Layer(("app.api", "app.cli"), True),
            Layer(("app.core", "app.util"), False),
        )

    def test_a_file_without_contracts_declares_none(self) -> None:
        reader = ImportLinterContracts()
        assert reader.read("pyproject.toml", "[project]\nname = 'x'\n") == ()
        assert reader.read("setup.cfg", "[metadata]\nname = x\n") == ()

    @pytest.mark.parametrize(
        ("path", "text"),
        [
            ("pyproject.toml", "[tool.importlinter\n"),
            (
                "pyproject.toml",
                "[[tool.importlinter.contracts]]\nname = 'x'\ntype = 'forbidden'\n"
                "source_modules = ['a']\n",
            ),
            ("pyproject.toml", "[[tool.importlinter.contracts]]\nname = 'no type'\n"),
            (".importlinter", "[importlinter:contract:1\n"),
        ],
    )
    def test_an_unreadable_declaration_raises(self, path: str, text: str) -> None:
        with pytest.raises(ContractFormatError, match=path):
            ImportLinterContracts().read(path, text)


# --- the change surface (TER-EVD-006) ----------------------------------------


@pytest.mark.req("TER-EVD-006")
class TestChangeSurface:
    def test_an_edit_or_write_whose_tool_failed_changed_nothing(
        self, shop: Path
    ) -> None:
        s = Script()
        pricing_task(s)
        failed = "<tool_use_error>String to replace not found in file.</tool_use_error>"
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))", output=failed)
        s.write(at("src/app/new.py"), "X = 1\n", output=failed)
        g = ground(s, shop)
        assert [e.path for e in g.edits.values()] == [PRICING]
        assert len(g.failed_edits) == 2
        assert "src/app/new.py" not in g.modules
        [surface] = analyse(s, shop).surfaces
        assert {e.path for e in surface.edits} == {PRICING}

    def test_every_task_has_a_surface_and_every_edit_is_placed(
        self, shop: Path
    ) -> None:
        s = Script()
        pricing_task(s)
        s.edit(at("tests/test_pricing.py"), "== 12", "== 12.0")
        s.edit(at(WEB), "request", "req")
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        s.edit(at("README.md"), "tiny", "small")
        s.write("/tmp/scratch.py", "print(1)\n")
        [surface] = analyse(s, shop).surfaces
        assert surface.basis is SeedBasis.NAMED
        assert surface.seeds == (PRICING,)
        assert surface.neighbours == (MODEL, CHECKOUT)
        assert surface.tests == ("tests/test_checkout.py", "tests/test_pricing.py")
        placed = {e.path: e.placement for e in surface.edits}
        assert placed == {
            PRICING: EditPlacement.INSIDE,
            "tests/test_pricing.py": EditPlacement.INSIDE,
            WEB: EditPlacement.EXPANSION,
            SUMMARY: EditPlacement.UNRELATED,
            "README.md": EditPlacement.UNRELATED,
            "/tmp/scratch.py": EditPlacement.OUTSIDE_REPOSITORY,
        }
        edits = [
            e for e in s.events if e.tool and e.tool.native_name in ("Edit", "Write")
        ]
        assert [e.event_id for e in surface.edits] == [
            e.id for e in edits if e.kind.value == "tool.requested"
        ]
        assert [e.path for e in surface.outside()] == [WEB, SUMMARY, "README.md"]

    @pytest.mark.parametrize(
        "prompt",
        [
            "Fix the rounding in src/app/domain/pricing.py",
            "Fix the rounding in app.domain.pricing",
            "Fix the rounding in price_with_tax",
        ],
    )
    def test_a_prompt_names_seeds_by_file_module_or_symbol(
        self, shop: Path, prompt: str
    ) -> None:
        s = Script()
        pricing_task(s, prompt)
        [surface] = analyse(s, shop).surfaces
        assert surface.seeds == (PRICING,) and surface.basis is SeedBasis.NAMED

    def test_a_prompt_that_acknowledges_inherits_the_named_seeds(
        self, shop: Path
    ) -> None:
        s = Script()
        first = s.prompt("Plan the rounding fix in pricing.py, do not edit yet")
        s.say("Plan: change the factor.")
        s.prompt("go ahead")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        [surface] = analyse(s, shop).surfaces
        assert surface.basis is SeedBasis.INHERITED
        assert surface.named_by == first.id and surface.seeds == (PRICING,)

    def test_with_nothing_named_the_first_edit_is_the_seed(self, shop: Path) -> None:
        s = Script()
        s.prompt("The monthly report counts wrong, please fix it")
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        [surface] = analyse(s, shop).surfaces
        assert surface.basis is SeedBasis.FIRST_EDIT and surface.named_by is None
        assert surface.seeds == (SUMMARY,) and surface.tests == (
            "tests/test_summary.py",
        )
        assert [e.placement for e in surface.edits] == [
            EditPlacement.INSIDE,
            EditPlacement.UNRELATED,
        ]

    def test_a_changed_intent_does_not_inherit(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.prompt(
            "Separate job: the monthly report totals are wrong, count every "
            "row in the report output"
        )
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        surfaces = analyse(s, shop).surfaces
        assert [x.basis for x in surfaces] == [SeedBasis.NAMED, SeedBasis.FIRST_EDIT]

    def test_created_files_linked_to_the_surface_are_inside(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.write(
            at("src/app/domain/rounding.py"),
            "from app.domain.pricing import price_with_tax\n",
        )
        [surface] = analyse(s, shop).surfaces
        assert surface.created == ("src/app/domain/rounding.py",)
        assert "src/app/domain/rounding.py" in surface.neighbours
        assert surface.edits[-1].placement is EditPlacement.INSIDE

    def test_a_created_test_of_the_seed_is_inside(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.write(
            at("tests/test_rounding.py"),
            "from app.domain.pricing import price_with_tax\n\n\n"
            "def test_round():\n    assert price_with_tax([5]) == 6\n",
        )
        s.write(
            at("src/app/reports/csv_export.py"), "def export(rows):\n    return rows\n"
        )
        [surface] = analyse(s, shop).surfaces
        assert "tests/test_rounding.py" in surface.tests
        assert surface.created == ("tests/test_rounding.py",)
        placed = {e.path: e.placement for e in surface.edits}
        assert placed["tests/test_rounding.py"] is EditPlacement.INSIDE
        assert placed["src/app/reports/csv_export.py"] is EditPlacement.UNRELATED

    def test_a_new_import_made_by_the_task_links_its_target(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.edit(
            at(PRICING),
            "from app.domain.model import order_total\n",
            "from app.domain.model import order_total\n"
            "from app.reports.summary import monthly_summary\n",
        )
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        [surface] = analyse(s, shop).surfaces
        assert SUMMARY in surface.neighbours
        assert all(e.placement is EditPlacement.INSIDE for e in surface.edits)

    def test_surfaces_are_reported_only_when_grounded(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        grounded = analyse(s, shop).as_dict(graph=False)
        plain = explain(s.events, RegexTokenizer()).as_dict(graph=False)
        assert grounded["change_surfaces"] and grounded["repository"]
        assert "change_surfaces" not in plain and "repository" not in plain
        assert GROUNDED.isdisjoint(d["id"] for d in plain["detectors"])  # type: ignore[union-attr]
        assert GROUNDED <= {d["id"] for d in grounded["detectors"]}  # type: ignore[union-attr]


@pytest.mark.req("TER-EVD-006")
class TestUnrelatedModification:
    def test_an_unlinked_python_edit_under_a_named_surface(self, shop: Path) -> None:
        s = Script()
        prompt = s.prompt("Fix the rounding in pricing.py")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        edit, result = s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert f.confidence == 0.6 and f.uncertain
        # Judged not waste (0 of 29): a risk that claims no cost (ADR 0006).
        assert f.waste is LeanWaste.OVERPRODUCTION and f.kind is FindingKind.RISK
        assert result is not None
        assert f.evidence == (prompt.id, edit.id, result.id)
        assert f.waste_events == () and f.tokens == 0
        assert f.subject == SUMMARY and SUMMARY in f.title

    def test_edits_inside_the_surface_are_not_reported(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.edit(at(MODEL), "sum(items)", "sum(items, 0)")
        s.edit(at("tests/test_checkout.py"), "== 12", "== 12.0")
        s.bash("pytest -q", PASS)
        a = analyse(s, shop)
        assert not [f for f in a.findings if f.detector in GROUNDED]

    def test_several_edits_of_one_file_are_one_finding(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        first, _ = s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        second, _ = s.edit(at(SUMMARY), "monthly_summary", "month_summary")
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert first.id in f.evidence and second.id in f.evidence
        assert f.id.startswith("unrelated_modification:")

    def test_a_new_test_module_with_no_link_is_uncertain(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.write(
            at("tests/test_cli_smoke.py"),
            "from app.cli import main\n\n\ndef test_main():\n    assert main([]) == 0\n",
        )
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert f.subject == "tests/test_cli_smoke.py"
        assert f.confidence == 0.55 and f.uncertain

    def test_an_existing_test_module_with_no_link_scores_as_any_file(
        self, shop: Path
    ) -> None:
        s = Script()
        pricing_task(s)
        s.write(at("src/app/reports/test_like.py"), "X = 1\n")
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        found_ = {
            f.subject: f for f in found(analyse(s, shop), "unrelated_modification")
        }
        assert found_[SUMMARY].confidence == 0.6

    def test_a_file_without_import_evidence_is_uncertain(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.edit(at("README.md"), "tiny", "small")
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert f.confidence == 0.55 and f.uncertain

    def test_an_engine_without_syntax_trees_is_uncertain(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        [f] = found(analyse(s, shop, engine="lexical"), "unrelated_modification")
        assert f.confidence == 0.55

    def test_an_inherited_surface_is_uncertain(self, shop: Path) -> None:
        s = Script()
        s.prompt("Plan the rounding fix in pricing.py, do not edit yet")
        s.say("Plan: change the factor.")
        s.prompt("go ahead")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert f.confidence == 0.65 and f.uncertain

    def test_a_first_edit_surface_is_uncertain(self, shop: Path) -> None:
        s = Script()
        s.prompt("The monthly report counts wrong, please fix it")
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        [f] = found(analyse(s, shop), "unrelated_modification")
        assert f.confidence == 0.55 and f.subject == PRICING

    def test_edits_outside_the_repository_are_not_judged(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.write("/tmp/scratch.py", "print(1)\n")
        s.edit("/home/dev/other/x.py", "a", "b")
        assert found(analyse(s, shop), "unrelated_modification") == []


@pytest.mark.req("TER-EVD-006")
class TestSurfaceExpansion:
    def test_one_import_link_beyond_the_surface_is_uncertain_expansion(
        self, shop: Path
    ) -> None:
        s = Script()
        pricing_task(s)
        edit, _ = s.edit(at(WEB), "request", "req")
        a = analyse(s, shop)
        [f] = found(a, "surface_expansion")
        assert f.confidence == 0.6 and f.uncertain and f.waste_events[0] == edit.id
        assert CHECKOUT in f.explanation
        assert found(a, "unrelated_modification") == []

    def test_expansion_from_an_unnamed_surface_is_less_certain(
        self, shop: Path
    ) -> None:
        s = Script()
        s.prompt("The checkout total is off, please fix it")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.edit(at(WEB), "request", "req")
        [f] = found(analyse(s, shop), "surface_expansion")
        assert f.confidence == 0.5

    def test_a_neighbour_is_not_expansion(self, shop: Path) -> None:
        s = Script()
        pricing_task(s)
        s.edit(at(CHECKOUT), "items", "basket")
        assert found(analyse(s, shop), "surface_expansion") == []


# --- architecture boundaries (TER-EVD-007) -----------------------------------


_UPWARD = "from app.service.checkout import checkout\n\n\nclass Order:"


@pytest.mark.req("TER-EVD-007")
class TestBoundaryViolation:
    def test_an_import_from_a_higher_layer_is_a_violation(self, shop: Path) -> None:
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        edit, result = s.edit(at(MODEL), "class Order:", _UPWARD)
        [f] = found(analyse(s, shop), "boundary_violation")
        assert f.kind is FindingKind.RISK and f.waste is LeanWaste.DEFECTS
        assert f.confidence == 0.9 and not f.uncertain
        assert result is not None and f.evidence == (edit.id, result.id)
        assert f.subject == "layers"
        assert "app.domain.model -> app.service.checkout" in f.title
        assert "line 1" in f.explanation
        assert f.id == f"boundary_violation:{edit.id}:layers:app.service.checkout"

    def test_a_forbidden_module_is_a_violation(self, shop: Path) -> None:
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        s.edit(at(MODEL), "class Order:", "import requests\n\n\nclass Order:")
        [f] = found(analyse(s, shop), "boundary_violation")
        assert f.subject == "pure-domain" and "requests" in f.title

    def test_an_allowed_import_is_not_a_violation(self, shop: Path) -> None:
        s = Script()
        s.prompt("Use the Order type in web.py")
        s.edit(
            at(WEB),
            "from app.service.checkout import checkout\n",
            "from app.domain.model import Order\n"
            "from app.service.checkout import checkout\n",
        )
        assert found(analyse(s, shop), "boundary_violation") == []

    def test_an_import_already_there_at_the_start_is_not_reported(
        self, tmp_path: Path
    ) -> None:
        files = dict(SHOP)
        files[MODEL] = "import requests\n\n\n" + SHOP[MODEL]
        root = shop_repo(tmp_path / "shop", files)
        s = Script()
        s.prompt("Rename order_total in model.py")
        s.edit(at(MODEL), "order_total", "order_sum")
        assert found(analyse(s, root), "boundary_violation") == []

    def test_an_import_removed_later_is_uncertain(self, shop: Path) -> None:
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        s.edit(at(MODEL), "class Order:", _UPWARD)
        undo, _ = s.edit(
            at(MODEL), "from app.service.checkout import checkout\n\n\n", ""
        )
        [f] = found(analyse(s, shop), "boundary_violation")
        assert f.confidence == 0.55 and f.uncertain and undo.id in f.evidence

    def test_a_later_edit_that_cannot_be_replayed_leaves_it_unknown(
        self, shop: Path
    ) -> None:
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        s.edit(at(MODEL), "class Order:", _UPWARD)
        s.edit(at(MODEL), "text that is not in the file", "x")
        [f] = found(analyse(s, shop), "boundary_violation")
        assert f.confidence == 0.7

    def test_a_created_module_is_judged_too(self, shop: Path) -> None:
        s = Script()
        s.prompt("Add a currency module to the domain")
        s.write(
            at("src/app/domain/currency.py"),
            "from app.adapters.web import handle\n\nEUR = 'EUR'\n",
        )
        [f] = found(analyse(s, shop), "boundary_violation")
        assert "app.domain.currency -> app.adapters.web" in f.title

    def test_without_declared_contracts_nothing_is_reported(self, shop: Path) -> None:
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        s.edit(at(MODEL), "class Order:", _UPWARD)
        assert found(analyse(s, shop, contracts=False), "boundary_violation") == []

    def test_unreadable_contracts_are_reported_not_guessed(
        self, tmp_path: Path
    ) -> None:
        files = dict(SHOP)
        files["pyproject.toml"] = "[tool.importlinter\n"
        root = shop_repo(tmp_path / "shop", files)
        s = Script()
        s.prompt("Add a currency to the Order in model.py")
        s.edit(at(MODEL), "class Order:", _UPWARD)
        g = ground(s, root)
        assert g.contracts == () and g.contract_problem is not None
        assert "pyproject.toml" in g.contract_problem
        a = explain(s.events, RegexTokenizer(), repository=g)
        assert found(a, "boundary_violation") == []


# --- grounding is computed once; the analysis stays a fold -------------------


@pytest.mark.req("TER-EVD-006")
@pytest.mark.req("TER-EVD-007")
class TestGroundedFold:
    def _session(self) -> Script:
        s = Script()
        pricing_task(s)
        s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        s.edit(at(MODEL), "class Order:", _UPWARD)
        s.bash("pytest -q", PASS)
        s.say("done")
        return s

    def test_incremental_equals_batch_with_repository_evidence(
        self, shop: Path
    ) -> None:
        s = self._session()
        g = ground(s, shop)
        engine = AnalysisEngine(RegexTokenizer())
        for event in s.events:
            engine.apply(event)
            engine.apply(event)  # redelivery changes nothing
        live = engine.explain(repository=g)
        batch = explain_batch(s.events, RegexTokenizer(), repository=g)
        assert live.as_dict() == batch.as_dict()
        assert found(batch, "unrelated_modification")
        assert found(batch, "boundary_violation")

    def test_grounding_is_the_same_wherever_the_repository_lives(
        self, tmp_path: Path
    ) -> None:
        s = self._session()
        one = ground(s, shop_repo(tmp_path / "a" / "shop"))
        two = ground(s, shop_repo(tmp_path / "b" / "elsewhere"))
        assert one.as_dict() == two.as_dict()
        assert one.root == "/home/dev/shop"

    def test_every_grounded_detector_has_a_countermeasure_and_follow_up(
        self, shop: Path
    ) -> None:
        s = self._session()
        s.prompt("Fix the rounding in pricing.py again")
        s.edit(at(WEB), "request", "req")
        a = analyse(s, shop)
        fired = {f.detector for f in a.findings} & GROUNDED
        assert fired == GROUNDED
        measures = build_countermeasures(a.findings, a.steps)
        by = {c.detector: c for c in measures if c.detector in GROUNDED}
        assert set(by) == GROUNDED
        assert any(x.kind is ActionKind.HOOK for x in by["boundary_violation"].actions)
        assert SUMMARY in by["unrelated_modification"].actions[1].text
        assert by["surface_expansion"].uncertain
        metrics = {
            u.how
            for u in follow_ups(a.findings, flow_efficiency=None, avoidable_share=0)
        }
        assert {f"findings[detector={d}]" for d in GROUNDED} <= metrics
