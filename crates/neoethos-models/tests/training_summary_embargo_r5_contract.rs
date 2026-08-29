use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{
    Expr, ExprBinary, ExprCall, ImplItemFn, ItemFn, ItemStruct, Member, Pat, Stmt, Type, UseTree,
};

const AUTHORITY_SOURCE_FILE_COUNT: usize = 81;
const SUMMARY_FILE: &str = "runtime/artifacts.rs";
const BAYESIAN_PRODUCER_FILE: &str = "statistical/bayesian_impl.rs";
const LINEAR_PRODUCER_FILE: &str = "statistical/linear_impl.rs";
const DEEP_PRODUCER_FILE: &str = "deep_models.rs";

struct ParsedSource {
    relative_path: String,
    ast: syn::File,
}

fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn collect_rs_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read source directory {}: {error}", directory.display()))
        .map(|entry| entry.expect("read source directory entry"))
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_type = entry
            .file_type()
            .unwrap_or_else(|error| panic!("read file type {}: {error}", entry.path().display()));
        if file_type.is_dir() {
            collect_rs_files(&entry.path(), files);
        } else if file_type.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "rs")
        {
            files.push(entry.path());
        }
    }
}

fn parse_authority_sources() -> Vec<ParsedSource> {
    let root = source_root()
        .canonicalize()
        .expect("canonicalize neoethos-models source root");
    let mut paths = Vec::new();
    collect_rs_files(&root, &mut paths);
    paths.sort();
    assert_eq!(
        paths.len(),
        AUTHORITY_SOURCE_FILE_COUNT,
        "recursive AST census drifted from the reviewed source set"
    );
    paths
        .into_iter()
        .map(|path| {
            let canonical = path
                .canonicalize()
                .unwrap_or_else(|error| panic!("canonicalize {}: {error}", path.display()));
            assert!(
                canonical.starts_with(&root),
                "source census escaped the authority root: {}",
                canonical.display()
            );
            let relative_path = canonical
                .strip_prefix(&root)
                .expect("census path is rooted below source")
                .to_string_lossy()
                .replace('\\', "/");
            let source = fs::read_to_string(&canonical)
                .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
            let ast = syn::parse_file(&source)
                .unwrap_or_else(|error| panic!("parse Rust AST for {relative_path}: {error}"));
            ParsedSource { relative_path, ast }
        })
        .collect()
}

#[derive(Default)]
struct SummaryStructVisitor<'ast> {
    matches: Vec<&'ast ItemStruct>,
}

impl<'ast> Visit<'ast> for SummaryStructVisitor<'ast> {
    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        if node.ident == "TrainingSummaryMetadata" {
            self.matches.push(node);
        }
        visit::visit_item_struct(self, node);
    }
}

fn type_is_usize(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Path(path)
            if path.qself.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].ident == "usize"
    )
}

#[test]
fn recursive_81_file_ast_census_requires_typed_four_way_summary() {
    let sources = parse_authority_sources();
    let mut declarations = Vec::new();
    for source in &sources {
        let mut visitor = SummaryStructVisitor::default();
        visitor.visit_file(&source.ast);
        declarations.extend(
            visitor
                .matches
                .into_iter()
                .map(|declaration| (source.relative_path.as_str(), declaration)),
        );
    }
    assert_eq!(
        declarations.len(),
        1,
        "TrainingSummaryMetadata must have exactly one typed authority declaration"
    );
    let (path, declaration) = declarations[0];
    assert_eq!(
        path, SUMMARY_FILE,
        "summary declaration moved without census review"
    );
    let fields = declaration
        .fields
        .iter()
        .map(|field| {
            let name = field
                .ident
                .as_ref()
                .expect("training-summary fields must be named")
                .to_string();
            assert!(type_is_usize(&field.ty), "{name} must be typed as usize");
            name
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        ["dataset_rows", "train_rows", "embargo_rows", "val_rows"],
        "typed training summary must preserve the temporal embargo explicitly"
    );
}

enum FunctionNode<'ast> {
    Free(&'ast ItemFn),
    Method(&'ast ImplItemFn),
}

impl FunctionNode<'_> {
    fn block(&self) -> &syn::Block {
        match self {
            Self::Free(function) => &function.block,
            Self::Method(function) => &function.block,
        }
    }

    fn signature(&self) -> &syn::Signature {
        match self {
            Self::Free(function) => &function.sig,
            Self::Method(function) => &function.sig,
        }
    }
}

#[derive(Default)]
struct NamedFunctionVisitor<'ast> {
    sought: &'static str,
    matches: Vec<FunctionNode<'ast>>,
}

impl<'ast> Visit<'ast> for NamedFunctionVisitor<'ast> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if node.sig.ident == self.sought {
            self.matches.push(FunctionNode::Free(node));
        }
        visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if node.sig.ident == self.sought {
            self.matches.push(FunctionNode::Method(node));
        }
        visit::visit_impl_item_fn(self, node);
    }
}

fn exact_function<'ast>(
    sources: &'ast [ParsedSource],
    relative_path: &str,
    function_name: &'static str,
) -> Result<FunctionNode<'ast>, String> {
    let source = sources
        .iter()
        .find(|source| source.relative_path == relative_path)
        .ok_or_else(|| format!("census did not contain exact producer file {relative_path}"))?;
    let mut visitor = NamedFunctionVisitor {
        sought: function_name,
        matches: Vec::new(),
    };
    visitor.visit_file(&source.ast);
    if visitor.matches.len() != 1 {
        return Err(format!(
            "{relative_path} must contain exactly one `{function_name}` producer, found {}",
            visitor.matches.len()
        ));
    }
    Ok(visitor.matches.pop().expect("one producer was checked"))
}

fn compact_tokens(value: &impl ToTokens) -> String {
    value
        .to_token_stream()
        .to_string()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[derive(Default)]
struct LocalBindingVisitor<'ast> {
    bindings: BTreeMap<String, &'ast Expr>,
}

impl<'ast> Visit<'ast> for LocalBindingVisitor<'ast> {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Pat::Ident(binding) = &local.pat
            && let Some(initializer) = &local.init
        {
            self.bindings
                .insert(binding.ident.to_string(), initializer.expr.as_ref());
        }
        visit::visit_local(self, local);
    }
}

fn resolve_local_expr<'ast>(
    mut expression: &'ast Expr,
    bindings: &BTreeMap<String, &'ast Expr>,
) -> &'ast Expr {
    let mut seen = BTreeSet::new();
    loop {
        let Expr::Path(path) = expression else {
            return expression;
        };
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return expression;
        }
        let name = path.path.segments[0].ident.to_string();
        if !seen.insert(name.clone()) {
            return expression;
        }
        let Some(resolved) = bindings.get(&name) else {
            return expression;
        };
        expression = resolved;
    }
}

#[derive(Default)]
struct CallVisitor<'ast> {
    calls: Vec<&'ast ExprCall>,
}

impl<'ast> Visit<'ast> for CallVisitor<'ast> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        self.calls.push(node);
        visit::visit_expr_call(self, node);
    }
}

fn path_segments(expression: &Expr) -> Option<Vec<String>> {
    match expression {
        Expr::Paren(parenthesized) => path_segments(&parenthesized.expr),
        Expr::Group(group) => path_segments(&group.expr),
        Expr::Path(path) => {
            let mut segments = Vec::new();
            if let Some(qself) = &path.qself
                && let Type::Path(qualified) = qself.ty.as_ref()
            {
                segments.extend(
                    qualified
                        .path
                        .segments
                        .iter()
                        .map(|segment| segment.ident.to_string()),
                );
            }
            segments.extend(
                path.path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string()),
            );
            Some(segments)
        }
        _ => None,
    }
}

fn producer_summary_expression<'ast>(
    function: &'ast FunctionNode<'ast>,
    summary_is_direct_return: bool,
) -> Result<(&'ast Expr, BTreeMap<String, &'ast Expr>), String> {
    let mut bindings = LocalBindingVisitor::default();
    bindings.visit_block(function.block());
    if summary_is_direct_return {
        let tail = function
            .block()
            .stmts
            .last()
            .and_then(|statement| match statement {
                Stmt::Expr(expression, None) => Some(expression),
                _ => None,
            })
            .ok_or_else(|| "producer has no tail summary expression".to_string())?;
        return Ok((
            resolve_local_expr(tail, &bindings.bindings),
            bindings.bindings,
        ));
    }

    let mut calls = CallVisitor::default();
    calls.visit_block(function.block());
    let metadata_calls = calls
        .calls
        .into_iter()
        .filter(|call| {
            let callee = resolve_local_expr(call.func.as_ref(), &bindings.bindings);
            path_segments(callee)
                .and_then(|segments| segments.last().cloned())
                .is_some_and(|name| name == "try_build_runtime_artifact_metadata")
        })
        .collect::<Vec<_>>();
    if metadata_calls.len() != 1 {
        return Err(format!(
            "producer must contain exactly one runtime-artifact builder call, found {}",
            metadata_calls.len()
        ));
    }
    let summary = metadata_calls[0]
        .args
        .last()
        .ok_or_else(|| "runtime-artifact builder has no summary argument".to_string())?;
    Ok((
        resolve_local_expr(summary, &bindings.bindings),
        bindings.bindings,
    ))
}

fn resolved_call_arguments(
    expression: &Expr,
    bindings: &BTreeMap<String, &Expr>,
) -> Result<Vec<String>, String> {
    let expression = resolve_local_expr(expression, bindings);
    let Expr::Call(call) = expression else {
        return Err(format!(
            "summary expression is not a constructor call: {}",
            compact_tokens(expression)
        ));
    };
    Ok(call
        .args
        .iter()
        .map(|argument| compact_tokens(resolve_local_expr(argument, bindings)))
        .collect())
}

struct ProducerSpec {
    file: &'static str,
    function: &'static str,
    direct_return: bool,
    requires_embargo_parameter: bool,
    expected_arguments: [&'static str; 4],
}

fn inspect_producer(sources: &[ParsedSource], spec: &ProducerSpec) -> Vec<String> {
    let mut errors = Vec::new();
    let function = match exact_function(sources, spec.file, spec.function) {
        Ok(function) => function,
        Err(error) => return vec![error],
    };
    let signature = compact_tokens(function.signature());
    if spec.requires_embargo_parameter && !signature.contains("embargo_rows:usize") {
        errors.push(format!(
            "typed embargo_rows parameter missing from signature `{signature}`"
        ));
    }
    match producer_summary_expression(&function, spec.direct_return) {
        Ok((expression, bindings)) => match resolved_call_arguments(expression, &bindings) {
            Ok(arguments) if arguments == spec.expected_arguments => {}
            Ok(arguments) => errors.push(format!(
                "expected four distinct {:?}, found {:?}",
                spec.expected_arguments, arguments
            )),
            Err(error) => errors.push(error),
        },
        Err(error) => errors.push(error),
    }
    errors
}

#[derive(Default)]
struct BinaryExpressionVisitor<'ast> {
    expressions: Vec<&'ast ExprBinary>,
}

impl<'ast> Visit<'ast> for BinaryExpressionVisitor<'ast> {
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        self.expressions.push(node);
        visit::visit_expr_binary(self, node);
    }
}

struct SummaryFieldVisitor<'bindings, 'ast> {
    bindings: &'bindings BTreeMap<String, &'ast Expr>,
    seen_bindings: BTreeSet<String>,
    fields: BTreeSet<String>,
}

impl<'bindings, 'ast> Visit<'ast> for SummaryFieldVisitor<'bindings, 'ast> {
    fn visit_expr_field(&mut self, node: &'ast syn::ExprField) {
        if let Member::Named(field) = &node.member
            && matches!(
                field.to_string().as_str(),
                "dataset_rows" | "train_rows" | "embargo_rows" | "val_rows"
            )
        {
            self.fields.insert(field.to_string());
        }
        visit::visit_expr_field(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if node.qself.is_none() && node.path.segments.len() == 1 {
            let name = node.path.segments[0].ident.to_string();
            if self.seen_bindings.insert(name.clone())
                && let Some(expression) = self.bindings.get(&name)
            {
                self.visit_expr(expression);
                return;
            }
        }
        visit::visit_expr_path(self, node);
    }
}

fn resolved_summary_fields_in_binary<'ast>(
    expression: &'ast ExprBinary,
    bindings: &BTreeMap<String, &'ast Expr>,
) -> BTreeSet<String> {
    let mut visitor = SummaryFieldVisitor {
        bindings,
        seen_bindings: BTreeSet::new(),
        fields: BTreeSet::new(),
    };
    visitor.visit_expr_binary(expression);
    visitor.fields
}

struct ValidatorSpec {
    file: &'static str,
    function: &'static str,
}

fn inspect_validator(sources: &[ParsedSource], spec: &ValidatorSpec) -> Vec<String> {
    let function = match exact_function(sources, spec.file, spec.function) {
        Ok(function) => function,
        Err(error) => return vec![error],
    };
    let mut bindings = LocalBindingVisitor::default();
    bindings.visit_block(function.block());
    let mut binary = BinaryExpressionVisitor::default();
    binary.visit_block(function.block());
    let required_without_embargo = ["dataset_rows", "train_rows", "val_rows"]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let comparisons = binary
        .expressions
        .into_iter()
        .filter_map(|expression| {
            let fields = resolved_summary_fields_in_binary(expression, &bindings.bindings);
            required_without_embargo
                .is_subset(&fields)
                .then_some((expression, fields))
        })
        .collect::<Vec<_>>();
    if comparisons.is_empty() {
        return vec!["no dataset/train/validation row-identity comparison found".to_string()];
    }
    comparisons
        .into_iter()
        .filter(|(_, fields)| !fields.contains("embargo_rows"))
        .map(|(expression, fields)| {
            format!(
                "row-identity comparison omits embargo_rows ({fields:?}): {}",
                compact_tokens(expression)
            )
        })
        .collect()
}

#[test]
fn all_three_real_summary_producer_failures_are_reported_together() {
    let sources = parse_authority_sources();
    let specs = [
        ProducerSpec {
            file: BAYESIAN_PRODUCER_FILE,
            function: "runtime_metadata",
            direct_return: false,
            requires_embargo_parameter: true,
            expected_arguments: ["dataset_rows", "train_rows", "embargo_rows", "val_rows"],
        },
        ProducerSpec {
            file: LINEAR_PRODUCER_FILE,
            function: "runtime_metadata",
            direct_return: false,
            requires_embargo_parameter: true,
            expected_arguments: ["dataset_rows", "train_rows", "embargo_rows", "val_rows"],
        },
        ProducerSpec {
            file: DEEP_PRODUCER_FILE,
            function: "training_summary_from_report",
            direct_return: true,
            requires_embargo_parameter: false,
            expected_arguments: [
                "report.dataset_rows",
                "report.train_rows",
                "report.embargo_rows",
                "report.val_rows",
            ],
        },
    ];
    let mut failures = Vec::new();
    for spec in &specs {
        let errors = inspect_producer(&sources, spec);
        if !errors.is_empty() {
            failures.push(format!(
                "{}::{}: {}",
                spec.file,
                spec.function,
                errors.join("; ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "all stale training-summary producers ({} of {}) were inspected:\n{}",
        failures.len(),
        specs.len(),
        failures.join("\n")
    );
}

#[test]
fn all_three_real_summary_validator_failures_are_reported_together() {
    let sources = parse_authority_sources();
    let specs = [
        ValidatorSpec {
            file: BAYESIAN_PRODUCER_FILE,
            function: "validate_runtime_metadata",
        },
        ValidatorSpec {
            file: LINEAR_PRODUCER_FILE,
            function: "validate_runtime_metadata",
        },
        ValidatorSpec {
            file: DEEP_PRODUCER_FILE,
            function: "validate_training_summary",
        },
    ];
    let failures = specs
        .iter()
        .filter_map(|spec| {
            let errors = inspect_validator(&sources, spec);
            (!errors.is_empty())
                .then(|| format!("{}::{}: {}", spec.file, spec.function, errors.join("; ")))
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "all stale training-summary validators ({} of {}) were inspected:\n{}",
        failures.len(),
        specs.len(),
        failures.join("\n")
    );
}

#[derive(Default)]
struct AliasTable {
    summary_types: BTreeSet<String>,
    summary_constructors: BTreeSet<String>,
}

impl AliasTable {
    fn from_file(file: &syn::File) -> Self {
        let mut table = Self::default();
        table
            .summary_types
            .insert("TrainingSummaryMetadata".to_string());
        let mut items = AliasItemVisitor::default();
        items.visit_file(file);
        let mut changed = true;
        while changed {
            let before = table.summary_types.len() + table.summary_constructors.len();
            for item_use in &items.uses {
                collect_use_aliases(&item_use.tree, &mut Vec::new(), &mut table);
            }
            for item_type in &items.types {
                if let Type::Path(path) = item_type.ty.as_ref()
                    && path.path.segments.last().is_some_and(|segment| {
                        table.summary_types.contains(&segment.ident.to_string())
                    })
                {
                    table.summary_types.insert(item_type.ident.to_string());
                }
            }
            changed = before != table.summary_types.len() + table.summary_constructors.len();
        }
        table
    }
}

#[derive(Default)]
struct AliasItemVisitor<'ast> {
    uses: Vec<&'ast syn::ItemUse>,
    types: Vec<&'ast syn::ItemType>,
}

impl<'ast> Visit<'ast> for AliasItemVisitor<'ast> {
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.uses.push(node);
        visit::visit_item_use(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        self.types.push(node);
        visit::visit_item_type(self, node);
    }
}

fn collect_use_aliases(tree: &UseTree, prefix: &mut Vec<String>, table: &mut AliasTable) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_aliases(path.tree.as_ref(), prefix, table);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let mut path = prefix.clone();
            path.push(name.ident.to_string());
            register_use_alias(&path, name.ident.to_string(), table);
        }
        UseTree::Rename(rename) => {
            let mut path = prefix.clone();
            path.push(rename.ident.to_string());
            register_use_alias(&path, rename.rename.to_string(), table);
        }
        UseTree::Group(group) => {
            for item in &group.items {
                collect_use_aliases(item, prefix, table);
            }
        }
        UseTree::Glob(_) => {}
    }
}

fn register_use_alias(path: &[String], alias: String, table: &mut AliasTable) {
    let refers_to_summary_type = path
        .iter()
        .any(|segment| table.summary_types.contains(segment));
    let refers_to_summary_constructor = path
        .iter()
        .any(|segment| table.summary_constructors.contains(segment));
    if refers_to_summary_type {
        if path.last().is_some_and(|segment| {
            matches!(
                segment.as_str(),
                "new" | "new_unchecked" | "raw_for_validation"
            )
        }) {
            table.summary_constructors.insert(alias);
        } else {
            table.summary_types.insert(alias);
        }
    } else if refers_to_summary_constructor {
        table.summary_constructors.insert(alias);
    }
}

fn is_summary_constructor(
    call: &ExprCall,
    table: &AliasTable,
    bindings: &BTreeMap<String, &Expr>,
    inside_summary_impl: bool,
) -> bool {
    let callee = resolve_local_expr(call.func.as_ref(), bindings);
    let Some(segments) = path_segments(callee) else {
        return false;
    };
    if segments.len() == 1 && table.summary_constructors.contains(&segments[0]) {
        return true;
    }
    let Some(constructor) = segments.last() else {
        return false;
    };
    if !matches!(
        constructor.as_str(),
        "new" | "new_unchecked" | "raw_for_validation"
    ) {
        return false;
    }
    if inside_summary_impl && segments.iter().any(|segment| segment == "Self") {
        return true;
    }
    segments
        .iter()
        .take(segments.len().saturating_sub(1))
        .any(|segment| table.summary_types.contains(segment))
}

struct InventoryCallVisitor<'ast, 'table> {
    table: &'table AliasTable,
    summary_impl_depth: usize,
    calls: Vec<(&'ast ExprCall, bool)>,
}

impl<'ast, 'table> Visit<'ast> for InventoryCallVisitor<'ast, 'table> {
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let is_summary_impl = matches!(
            node.self_ty.as_ref(),
            Type::Path(path)
                if path.path.segments.last().is_some_and(|segment| {
                    self.table.summary_types.contains(&segment.ident.to_string())
                })
        );
        self.summary_impl_depth += usize::from(is_summary_impl);
        visit::visit_item_impl(self, node);
        self.summary_impl_depth -= usize::from(is_summary_impl);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        self.calls.push((node, self.summary_impl_depth > 0));
        visit::visit_expr_call(self, node);
    }
}

fn alias_aware_constructor_calls(file: &syn::File) -> Vec<&ExprCall> {
    let table = AliasTable::from_file(file);
    let mut bindings = LocalBindingVisitor::default();
    bindings.visit_file(file);
    let mut calls = InventoryCallVisitor {
        table: &table,
        summary_impl_depth: 0,
        calls: Vec::new(),
    };
    calls.visit_file(file);
    calls
        .calls
        .into_iter()
        .filter(|(call, inside_summary_impl)| {
            is_summary_constructor(call, &table, &bindings.bindings, *inside_summary_impl)
        })
        .map(|(call, _)| call)
        .collect()
}

#[test]
fn alias_resolver_covers_use_type_and_local_constructor_aliases() {
    let source = syn::parse_file(
        r#"
        use crate::runtime::artifacts::TrainingSummaryMetadata as Imported;
        use crate::runtime::artifacts::TrainingSummaryMetadata::new as imported_new;
        use Imported as ImportedAgain;
        use ImportedAgain::new_unchecked as chained_new;
        type Typed = Imported;
        fn sample(a: usize, b: usize, c: usize, d: usize) {
            let local_new = Typed::new_unchecked;
            let parenthesized = (Typed::raw_for_validation);
            let _ = Imported::new(a, b, c, d);
            let _ = imported_new(a, b, c, d);
            let _ = local_new(a, b, c, d);
            let _ = chained_new(a, b, c, d);
            let _ = parenthesized(a, b, c, d);
            let _ = <Typed>::new_unchecked(a, b, c, d);
        }
        impl Imported {
            fn self_alias(a: usize, b: usize, c: usize, d: usize) {
                let _ = Self::new(a, b, c, d);
            }
        }
        mod nested {
            use crate::runtime::artifacts::TrainingSummaryMetadata as Nested;
            fn sample(a: usize, b: usize, c: usize, d: usize) {
                let local = Nested::new;
                let _ = local(a, b, c, d);
            }
        }
        "#,
    )
    .expect("parse alias-resistance fixture");
    let calls = alias_aware_constructor_calls(&source);
    assert_eq!(
        calls.len(),
        8,
        "every constructor alias must enter the census"
    );
}

#[test]
fn validator_census_resolves_local_row_count_aliases() {
    let parsed = |body: &str| ParsedSource {
        relative_path: "synthetic.rs".to_string(),
        ast: syn::parse_file(body).expect("parse validator alias fixture"),
    };
    let four_way = parsed(
        r#"
        fn validate(summary: &TrainingSummaryMetadata) {
            let training = summary.train_rows;
            let held_out = summary.embargo_rows;
            let validation = summary.val_rows;
            let observed = training + held_out + validation;
            if summary.dataset_rows != observed {}
        }
        "#,
    );
    let spec = ValidatorSpec {
        file: "synthetic.rs",
        function: "validate",
    };
    assert!(
        inspect_validator(&[four_way], &spec).is_empty(),
        "four-way row identity hidden behind aliases must pass"
    );

    let three_way = parsed(
        r#"
        fn validate(summary: &TrainingSummaryMetadata) {
            let training = summary.train_rows;
            let validation = summary.val_rows;
            let observed = training + validation;
            if summary.dataset_rows != observed {}
        }
        "#,
    );
    let errors = inspect_validator(&[three_way], &spec);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("omits embargo_rows"));
}

#[test]
fn alias_aware_constructor_inventory_has_no_three_way_embargo_collapse() {
    let sources = parse_authority_sources();
    let mut stale = Vec::new();
    let mut total = 0usize;
    for source in &sources {
        for call in alias_aware_constructor_calls(&source.ast) {
            total += 1;
            if call.args.len() != 4 {
                stale.push(format!(
                    "{}: {}",
                    source.relative_path,
                    compact_tokens(call)
                ));
            }
        }
    }
    assert!(total > 0, "typed constructor inventory must be non-empty");
    assert!(
        stale.is_empty(),
        "three-way TrainingSummaryMetadata constructors remain ({}):\n{}",
        stale.len(),
        stale.join("\n")
    );
}
